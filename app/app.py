from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import logging 
from utils import preprocess_text, extract_phobert_features_single

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI()

# Load model và scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục chứa app.py
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_variant_2.json")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)
# Định nghĩa request body
class TextRequest(BaseModel):
    text: str

# Định nghĩa response body
class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

# Endpoint chào mừng
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Endpoint để dự đoán
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        logger.info(f"Received text: {request.text}")

        processed_text = preprocess_text(request.text)
        features = extract_phobert_features_single(processed_text)
        #print(f"Extracted features: {features}")
        features_scaled = scaler.transform([features])
        
        prediction = model.predict(features_scaled)
        confidence = np.max(model.predict_proba(features_scaled))

        logger.info(f"Feature: {features_scaled}, Prediction: {prediction[0]}, Confidence: {confidence}")

        return PredictionResponse(prediction=int(prediction[0]), confidence=float(confidence))
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)