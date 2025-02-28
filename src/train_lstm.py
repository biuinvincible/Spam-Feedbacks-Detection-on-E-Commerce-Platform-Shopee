import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from utils import EarlyStopping

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Xác định thư mục gốc của dự án
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Một cấp trên thư mục chứa file hiện tại
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")  
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

# Load config
config_path = os.path.join(BASE_DIR, "configs", "lstm.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load preprocessed data
data_dir = os.path.join(BASE_DIR, "data", "processed")
X_train = np.load(os.path.join(data_dir, "lstm_X_train.npy"))
X_test = np.load(os.path.join(data_dir, "lstm_X_test.npy"))
y_train = np.load(os.path.join(data_dir, "lstm_y_train.npy"))
y_test = np.load(os.path.join(data_dir, "lstm_y_test.npy"))
vocab_info = np.load(os.path.join(data_dir, "lstm_vocab_info.npy"), allow_pickle=True).item()

vocab_size = vocab_info["vocab_size"]
pad_idx = vocab_info["pad_idx"]

# Chuyển dữ liệu thành tensor và đưa vào thiết bị
X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Model LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden_last = hidden[-1]
        out = self.dropout(hidden_last)
        out = self.fc(out)
        return self.sigmoid(out)

# Lặp qua tất cả model_variants
for variant_name, params in config["model_variants"].items():
    print(f"Training model: {variant_name} with params: {params}")
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=params["batch_size"], shuffle=False)

    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=params.get("embedding_dim", 300),
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        pad_idx=pad_idx
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    early_stopping = EarlyStopping(patience=params.get("patience", 5), verbose=True)

    # Training loop
    mlflow.set_experiment("shopee_spam_lstm")
    with mlflow.start_run(run_name=variant_name):
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
            
            # Evaluation
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
            
            # Log metrics
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("precision", precision, step=epoch)
            mlflow.log_metric("recall", recall, step=epoch)
            mlflow.log_metric("f1", f1, step=epoch)
            
            print(f"[{variant_name}] Epoch {epoch+1}/{params['epochs']} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
            
            # Early stopping
            model_path = os.path.join(BASE_DIR, "models", f"lstm_{variant_name}.pt")
            early_stopping(val_loss, model, model_path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load best model for final logging
        model.load_state_dict(torch.load(model_path))
        input_example = X_train_tensor[:1].cpu().numpy()
        model.to("cpu")
        mlflow.pytorch.log_model(model, f"lstm_model_{variant_name}", 
            pip_requirements=os.path.join(BASE_DIR, "requirements.txt"), 
            input_example=input_example)
        model.to(device)

print("Training complete for all model variants!")
