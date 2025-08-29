# file: api_logistic.py
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Define request schema
class InputData(BaseModel):
    features: list[float]   # must be 30 numbers

# Create FastAPI app
app = FastAPI(title="Logistic Regression API with MLflow")

# Load model once at startup
mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.sklearn.load_model("models:/logistic-classifier-model/1")

@app.post("/predict")
def predict(data: InputData):
    # Convert to numpy array
    X = np.array([data.features])

    # Prediction
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()

    return {
        "prediction": int(pred),
        "probabilities": {
            "benign": proba[0],
            "malignant": proba[1]
        }
    }
