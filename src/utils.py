import os
import re
import numpy as np
from transformers import AutoModel, PhobertTokenizer
from nltk import word_tokenize
import torch
import py_vncorenlp
from py_vncorenlp import VnCoreNLP
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

# Class for early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait before stopping after loss improvement.
            min_delta (float): Minimum change in loss to qualify as an improvement.
            verbose (bool): If True, prints message for each loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = float('inf')

    def __call__(self, val_loss, model, model_path):
        """
        Args:
            val_loss (float): Current validation loss
            model: PyTorch model to save if validation loss improves
            model_path (str): Path to save the model
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, model_path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.min_validation_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), model_path)
        self.min_validation_loss = val_loss

def load_vncorenlp():
    # Load VnCoreNLP models
    VNCORP_PATH = os.path.join(os.path.dirname(__file__), "VnCoreNLP-master")
    # Tải model (nếu cần)
    py_vncorenlp.download_model(save_dir=VNCORP_PATH)
    # Load model
    model = py_vncorenlp.VnCoreNLP(save_dir=VNCORP_PATH)
    return model

def load_bert():
    v_phobert = AutoModel.from_pretrained('vinai/phobert-base')
    v_tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

def standardize_data(row):
    if not isinstance(row, str):
        text = str(row)

    # Define the pattern to match unwanted characters
    pattern = r'[^a-zA-ZÀ-Ỹà-ỹ\s]'  # Keep letters, digits, spaces, and Vietnamese characters

    # Use regex to substitute unwanted characters with an empty string
    cleaned_text = re.sub(pattern, ' ', row)

    return str(cleaned_text.lower())

def remove_repetitive_characters(text):
    # Define regex pattern to match repetitive characters
    pattern = re.sub(r'(.)\1+', r'\1\1', text)  # Match one character followed by one or more occurrences of the same character
    return pattern

def correct_spelling_teencode(text):
    # Dictionary of common teencode and their correct forms
    teencode_dict = {
        'chx': 'chưa',
        'z': 'vậy',
        'd': 'vậy',
        'k': 'không',
        'hok': 'không',
        'ko': 'không',
        'kh': 'không',
        'cx': 'cũng',
        'đỉm': 'điểm',
        'oce': 'ok',
        'oke': 'ok',
        'đc': 'được',
        'ns': 'nói',
        'tc': 'tính chất',
        'tch': 'tính chất',
        'tks': 'cảm ơn',
        'nc': 'nói chuyện',
        'thui': 'thôi',
        'ha': 'hình ảnh',
        'ik': 'đi',
        'auce': 'ok',
        'xink': 'xinh',
        'dth': 'dễ thương',
        'dthw': 'dễ thương',
        'nhe':'nha',
        'nthe': 'như thế',
        'dethun': 'dễ thương',
        'kcj': 'không có gì',
        'kcgi': 'không có gì',
        'ntn': 'như thế này',
        'ng': 'người',
        'mn': 'mọi người',
        'ng': 'mọi người',
        'nma': 'nhưng mà',
        'qlai': 'quay lại',
        'sp': 'sản phẩm',
        'tn': 'tin nhắn',
        'qtam': 'quan tâm',
        'th': 'thôi',
        'nch': 'nói chung',
        'mk': 'mình'

         # Add more teencode mappings as needed
    }

    # Tokenize the text into words
    words = word_tokenize(text)

    # Replace teencode with correct forms
    corrected_words = [teencode_dict[word] if word in teencode_dict else word for word in words]

    # Join the corrected words back into a single string
    corrected_text = ' '.join(corrected_words)
    return corrected_text

def extract_phobert_features(df, text_column, label_column=None, max_len=50, batch_size=2048):
    """
    Process text data through preprocessing, PhoBERT tokenization, and feature extraction
    with memory-efficient batch processing
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing text data
    text_column : str
        Name of the column containing text to process
    label_column : str, optional
        Name of the column containing labels
    max_len : int, default=100
        Maximum sequence length for tokenization
    batch_size : int, default=32
        Batch size for GPU processing. Reduced to prevent VRAM overflow.
        
    Returns:
    --------
    tuple
        (features array, labels array if label_column provided)
    """
    # Preprocessing
    df = df.copy()  # Create a copy to avoid modifying original dataframe
    # Get data
    texts = df[text_column].values
    y = df[label_column].values if label_column else None
    
    # Load BERT model and tokenizer
    v_phobert, v_tokenizer = load_bert()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    v_phobert = v_phobert.to(device)
    
    # Process in smaller chunks to avoid memory issues
    chunk_size = 1000  # Process 1000 texts at a time
    all_features = []
    
    for chunk_start in range(0, len(texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]
        
        # Word segmentation for current chunk
        chunk_texts = [model.word_segment(text) for text in chunk_texts]
        chunk_texts = [' '.join(words) for words in chunk_texts]  # Join words for tokenizer
        
        # Tokenize current chunk
        encodings = v_tokenizer(
            chunk_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        
        # Process chunk in batches
        chunk_features = []
        for i in range(0, len(chunk_texts), batch_size):
            batch_end = min(i + batch_size, len(chunk_texts))
            batch_input_ids = encodings['input_ids'][i:batch_end].to(device)
            batch_attention_mask = encodings['attention_mask'][i:batch_end].to(device)
            
            with torch.no_grad():
                outputs = v_phobert(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                # Get the [CLS] token embeddings
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                chunk_features.append(batch_features)
                
            # Clear GPU memory
            del batch_input_ids
            del batch_attention_mask
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Combine batch features for current chunk
        chunk_features = np.concatenate(chunk_features, axis=0)
        all_features.append(chunk_features)
        
        print(f"Processed texts {chunk_start} to {chunk_end} out of {len(texts)}")
    
    # Combine all chunk features
    v_features = np.concatenate(all_features, axis=0)
    print(f"Final features shape: {v_features.shape}")
    
    return (v_features, y) if label_column else v_features