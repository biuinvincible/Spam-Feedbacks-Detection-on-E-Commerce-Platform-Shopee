# Shopee Spam Detection

This project aims to classify spam comments in Shopee product reviews using multiple deep learning models (CNN, GRU, LSTM) and traditional machine learning (XGBoost). The training process is managed using MLflow for experiment tracking.

## Project Structure
```
Shopee_Spam_Detection/
│── configs/            # YAML configuration files for different models
│   ├── cnn.yaml        # Config for CNN model
│   ├── gru.yaml        # Config for GRU model
│   ├── lstm.yaml       # Config for LSTM model
│   ├── xgboost.yaml    # Config for XGBoost model
│── data/               # Raw and processed datasets
│── models/             # Saved trained models
│── src/                # Source code
│   ├── train_cnn.py    # Train CNN model
│   ├── train_gru.py    # Train GRU model
│   ├── train_lstm.py   # Train LSTM model
│   ├── train_xgboost.py # Train XGBoost model
│   ├── preprocessing.py # Preprocess the dataset
│── pipeline.py         # Runs all training scripts sequentially
│── requirements.txt    # Required dependencies
│── Dockerfile          # Docker container setup
│── README.md           # Project documentation
│── app/                # FastAPI application
│   ├── app.py          # FastAPI server
│   ├── utils.py        # Helper functions
│   ├── Dockerfile      # Docker setup for API
│   ├── requirements.txt # Dependencies for API
```

## Setup
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python src/preprocessing.py
```

### 3. Run Training Pipeline
```bash
python pipeline.py
```

### 4. Run FastAPI Application
#### a) Run locally
```bash
cd app
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Then open `http://localhost:8000/docs` to access the API documentation.

#### b) Run with Docker
```bash
cd app
docker build -t shopee_spam_api .
docker run -p 8000:8000 shopee_spam_api
```

## Tracking with MLflow
To monitor experiment results:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Then open `http://localhost:5000` in your browser.

## Contact
For any questions, feel free to ask me via https://www.facebook.com/anh.khoa.468258/.

