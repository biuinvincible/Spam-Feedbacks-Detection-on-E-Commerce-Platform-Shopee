import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from utils import EarlyStopping
from pathlib import Path
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# Định nghĩa đường dẫn thư mục gốc
BASE_DIR = str(Path(__file__).resolve().parent.parent)  # Chuyển BASE_DIR thành str
CONFIG_PATH = str(Path(BASE_DIR) / "configs" / "cnn.yaml")
DATA_PATH = Path(BASE_DIR) / "data" / "processed"
MODEL_PATH = Path(BASE_DIR) / "models"
REQUIREMENTS_PATH = str(Path(BASE_DIR) / "requirements.txt")
MLRUNS_DIR = str(Path(BASE_DIR) / "mlruns")
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

# Kiểm tra và set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load config
with open(CONFIG_PATH, "r") as f:
    config_cnn = yaml.safe_load(f)

# Load preprocessed data
X_train = np.load(DATA_PATH / "cnn_X_train.npy")
X_test = np.load(DATA_PATH / "cnn_X_test.npy")
y_train = np.load(DATA_PATH / "cnn_y_train.npy")
y_test = np.load(DATA_PATH / "cnn_y_test.npy")
vocab_info = np.load(DATA_PATH / "cnn_vocab_info.npy", allow_pickle=True).item()
vocab_size = vocab_info["vocab_size"]
pad_idx = vocab_info["pad_idx"]

# Convert to tensor và chuyển sang device
X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, dropout, pad_idx):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim), padding=(fs // 2, 0)) 
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        convs = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.adaptive_max_pool1d(conv, 1).squeeze(2) for conv in convs]
        cat = torch.cat(pooled, dim=1)
        out = self.dropout(cat)
        out = self.fc(out)
        return self.sigmoid(out)

# Train function
def train_model(model, params, model_name):
    model = model.to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                            batch_size=params["batch_size"], 
                            shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), 
                           batch_size=params["batch_size"], 
                           shuffle=False)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    early_stopping = EarlyStopping(patience=params.get("patience", 5), verbose=True)
    
    mlflow.set_experiment(f"shopee_spam_{model_name}")
    
    with mlflow.start_run():
        mlflow.log_params(params)
        
        for epoch in range(params["epochs"]):
            model.train()
            total_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                y_preds = []
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch).squeeze()
                    val_loss += criterion(y_pred, y_batch).item()
                    y_preds.extend((y_pred > 0.5).cpu().numpy())
                
                val_loss = val_loss / len(test_loader)
                accuracy = accuracy_score(y_test, y_preds)
                precision = precision_score(y_test, y_preds)
                recall = recall_score(y_test, y_preds)
                f1 = f1_score(y_test, y_preds)
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("precision", precision, step=epoch)
            mlflow.log_metric("recall", recall, step=epoch)
            mlflow.log_metric("f1", f1, step=epoch)
            
            print(f"Epoch {epoch+1}/{params['epochs']} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
            
            # Early stopping
            model_save_path = MODEL_PATH / f"{model_name}_model.pt"
            early_stopping(val_loss, model, model_save_path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Load best model for final logging
        model.load_state_dict(torch.load(model_save_path))
        input_example = X_train_tensor[:1].cpu().numpy()
        model.to("cpu")
        mlflow.pytorch.log_model(model, f"{model_name}_model", input_example=input_example, pip_requirements=REQUIREMENTS_PATH)
        model.to(device)

# Train all model variants
for variant_name, params in config_cnn["model_variants"].items():
    print(f"Starting training for {variant_name}...")
    cnn_model = CNNModel(vocab_size, params["embedding_dim"], params["num_filters"], 
                        params["filter_sizes"], params["dropout"], pad_idx)
    train_model(cnn_model, params, f"cnn_{variant_name}")
    print(f"Training complete for {variant_name}!")
