import pandas as pd
import yaml
import xgboost as xgb
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# Load config
with open("/mnt/d/shopee_spam_detection/configs/xgboost.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data
X_train = np.load("/mnt/d/shopee_spam_detection/data/processed/xgboost_X_train.npy")
X_test = np.load("/mnt/d/shopee_spam_detection/data/processed/xgboost_X_test.npy")
y_train = np.load("/mnt/d/shopee_spam_detection/data/processed/xgboost_y_train.npy")
y_test = np.load("/mnt/d/shopee_spam_detection/data/processed/xgboost_y_test.npy")

# Chuẩn hóa dữ liệu với MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bắt đầu logging với MLflow
mlflow.set_experiment("xgboost_experiment")

# Chạy tất cả model_variants
for variant_name, params in config["model_variants"].items():
    with mlflow.start_run(run_name=variant_name):
        print(f"Training model: {variant_name}")
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Dự đoán và đánh giá
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log kết quả vào MLflow
        input_example = X_train[:1]
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.xgboost.log_model(model, f"xgboost_model_{variant_name}", input_example=input_example, pip_requirements="/mnt/d/shopee_spam_detection/requirements.txt")
        
        # Lưu mô hình ra file
        model.save_model(f"/mnt/d/shopee_spam_detection/models/xgboost_{variant_name}.json")
        
        print(f"Accuracy for {variant_name}: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
