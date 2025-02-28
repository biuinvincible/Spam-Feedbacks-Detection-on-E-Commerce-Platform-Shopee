import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LRScheduler
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from py_vncorenlp import VnCoreNLP
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
from utils import (
    remove_repetitive_characters, 
    correct_spelling_teencode, 
    standardize_data,
    load_vncorenlp,
    EarlyStopping
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "labeled_data.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def preprocess_data(input_path, max_length=128, force_preprocess=False):
    """
    Preprocess data specifically for PhoBERT model
    
    Args:
        input_path: Path to the input CSV file
        output_dir: Directory to save processed data (if None, don't save)
        max_length: Maximum sequence length for tokenization
        force_preprocess: Whether to force preprocessing even if processed data exists
    
    Returns:
        Processed training and testing data
    """
    # Check if preprocessed data already exists
    if not force_preprocess:
        try:
            train_input_ids = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_train_input_ids.npy"))
            train_attention_masks = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_train_attention_masks.npy"))
            train_labels = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_train_labels.npy"))
            test_input_ids = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_test_input_ids.npy"))
            test_attention_masks = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_test_attention_masks.npy"))
            test_labels = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_test_labels.npy"))
            
            print("Loaded preprocessed data from disk.")
            return (train_input_ids, train_attention_masks, train_labels), (test_input_ids, test_attention_masks, test_labels)
        except (FileNotFoundError, IOError):
            print("Preprocessed data not found. Performing preprocessing...")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Convert sentiment to binary labels
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == "Spam" else 0)
    
    # Basic preprocessing
    print("Applying basic text preprocessing...")
    df["processed_comment"] = df["comment"].apply(lambda x: standardize_data(x))
    df["processed_comment"] = df["processed_comment"].apply(lambda x: correct_spelling_teencode(x))
    df["processed_comment"] = df["processed_comment"].apply(lambda x: remove_repetitive_characters(x))
    df = df[df["processed_comment"] != ""]
    
    # Initialize VnCoreNLP tokenizer
    print("Initializing VnCoreNLP tokenizer...")
    model = load_vncorenlp()
    
    # Initialize PhoBERT tokenizer
    print("Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Tokenize using VnCoreNLP and then PhoBERT
    def tokenize_text(text):
        # Segment words using VnCoreNLP
        try:
            segmented_text = ' '.join([' '.join(sent) for sent in model.word_segment(text)])
        except Exception as e:
            print(f"Error tokenizing text: {text}, error: {e}")
            segmented_text = text
            
        # Tokenize using PhoBERT tokenizer
        encoded = tokenizer(
            segmented_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {    
            'input_ids': encoded['input_ids'].squeeze().numpy(),
            'attention_mask': encoded['attention_mask'].squeeze().numpy()
        }
    
    print("Tokenizing data...")
    # Apply tokenization
    tokenized_data = []
    for text in tqdm(df["processed_comment"], desc="Tokenizing"):
        tokenized_data.append(tokenize_text(text))
    
    # Extract input_ids and attention_mask
    input_ids = np.array([item['input_ids'] for item in tokenized_data])
    attention_masks = np.array([item['attention_mask'] for item in tokenized_data])
    labels = df["label"].values
    
    # Split data
    print("Splitting data into train and test sets...")
    train_ids, test_ids, train_masks, test_masks, train_labels, test_labels = train_test_split(
        input_ids, attention_masks, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Save processed data if output_dir is provided
    if PROCESSED_DATA_DIR:
        # Create output directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Save processed data
        print("Saving processed data...")
        np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_train_input_ids.npy"), train_ids)
        np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_train_attention_masks.npy"), train_masks)
        np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_train_labels.npy"), train_labels)
        np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_test_input_ids.npy"), test_ids)
        np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_test_attention_masks.npy"), test_masks)
        np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_test_labels.npy"), test_labels)
        
        # Save tokenizer info
        tokenizer_info = {
            'max_length': max_length
        }
        np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_tokenizer_info.npy"), tokenizer_info)
    
    print("Preprocessing completed!")
    return (train_ids, train_masks, train_labels), (test_ids, test_masks, test_labels)

def create_dataloaders(train_data, test_data, batch_size):
    """Create PyTorch DataLoaders for training and testing"""
    train_input_ids, train_attention_masks, train_labels = train_data
    test_input_ids, test_attention_masks, test_labels = test_data
    
    # Convert to PyTorch tensors
    train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
    train_attention_masks = torch.tensor(train_attention_masks, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
    test_attention_masks = torch.tensor(test_attention_masks, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    return train_dataloader, test_dataloader

def train(model, dataloader, optimizer, scheduler, device, epoch=0):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total_correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    # Log metrics to MLflow
    mlflow.log_metric("train_loss", avg_loss, step=epoch)
    mlflow.log_metric("train_accuracy", accuracy, step=epoch)
    mlflow.log_metric("train_f1", f1, step=epoch)
    mlflow.log_metric("train_precision", precision, step=epoch)
    mlflow.log_metric("train_recall", recall, step=epoch)
    
    return avg_loss, accuracy, f1, precision, recall

def evaluate(model, dataloader, device, epoch=0):
    """Evaluate the model on the test set"""
    model.eval()
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    # Log metrics to MLflow
    mlflow.log_metric("val_loss", avg_loss, step=epoch)
    mlflow.log_metric("val_accuracy", accuracy, step=epoch)
    mlflow.log_metric("val_f1", f1, step=epoch)
    mlflow.log_metric("val_precision", precision, step=epoch)
    mlflow.log_metric("val_recall", recall, step=epoch)
    
    return avg_loss, accuracy, f1, precision, recall

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--model_name", type=str, default="phobert_base", help="Name for the experiment run")
    args = parser.parse_args()
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Preprocess data
    train_data, test_data = preprocess_data(DATA_DIR, force_preprocess=args.force_preprocess)
    
    # Create DataLoaders
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data, batch_size=args.batch_size)
    
    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    # Initialize early stopping
    model_save_path = os.path.join(BASE_DIR, "models", "phobert_binary_classifier.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Set up MLflow experiment
    mlflow.set_experiment("shopee_spam_phobert")
    
    # Start MLflow run
    with mlflow.start_run(run_name=args.model_name):
        # Log parameters
        params = {
            "model": "vinai/phobert-base",
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "patience": args.patience,
            "warmup_steps_ratio": 0.1,
            "optimizer": "AdamW",
            "scheduler": "LinearSchedulerWithWarmup"
        }
        mlflow.log_params(params)
        
        # Training loop
        best_f1 = 0
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            
            # Train the model
            train_loss, train_acc, train_f1, train_precision, train_recall = train(
                model, train_dataloader, optimizer, scheduler, device, epoch
            )
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
            
            # Evaluate the model
            val_loss, val_acc, val_f1, val_precision, val_recall = evaluate(
                model, test_dataloader, device, epoch
            )
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            print(f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
            
            # Early stopping
            early_stopping(val_loss, model, model_save_path)
            
            # Save best model based on F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_path = os.path.join(BASE_DIR, "models", f"phobert_best_f1.pth")
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(best_model_path)
                
                # Log best metrics
                mlflow.log_metric("best_val_f1", val_f1)
                mlflow.log_metric("best_val_accuracy", val_acc)
                mlflow.log_metric("best_val_precision", val_precision)
                mlflow.log_metric("best_val_recall", val_recall)
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(model_save_path))
        final_loss, final_acc, final_f1, final_precision, final_recall = evaluate(
            model, test_dataloader, device
        )
        
        # Log final metrics
        mlflow.log_metric("final_val_loss", final_loss)
        mlflow.log_metric("final_val_accuracy", final_acc)
        mlflow.log_metric("final_val_f1", final_f1)
        mlflow.log_metric("final_val_precision", final_precision)
        mlflow.log_metric("final_val_recall", final_recall)
        
        # Save model to MLflow - FIX FOR INPUT EXAMPLE ERROR
        model.cpu()
        
        # Create a proper input example using numpy arrays instead of lists/dicts
        # Take the first example from training data and convert to numpy arrays
        input_ids_np = train_data[0][0:1]  # Take first sample as numpy array
        attention_mask_np = train_data[1][0:1]  # Take first sample as numpy array
        
        # Create a pandas DataFrame for the example input (MLflow PyTorch flavor supports DataFrame)
        sample_df = pd.DataFrame({
            'input_ids': [input_ids_np[0].tolist()],  # Convert array to list for DataFrame
            'attention_mask': [attention_mask_np[0].tolist()]  # Convert array to list for DataFrame
        })
        
        # Define a signature for the model
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec
        
        input_schema = Schema([
            TensorSpec(np.dtype(np.int64), (-1, 128), name="input_ids"),
            TensorSpec(np.dtype(np.int64), (-1, 128), name="attention_mask")
        ])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 2))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Log the model with signature instead of input_example
        mlflow.pytorch.log_model(
            model, 
            "phobert_model",
            pip_requirements=os.path.join(BASE_DIR, "requirements.txt"),
            signature=signature  # Use signature instead of input_example
        )
        
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved to {model_save_path}")
        print("Training completed!")

if __name__ == "__main__":
    main()