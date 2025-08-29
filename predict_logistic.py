import mlflow
import mlflow.sklearn
import numpy as np

def predict_logistic():
    # Point to your MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5000")

    # Load model from Model Registry: adjust version as needed
    model = mlflow.sklearn.load_model("models:/logistic-classifier-model/1")

    # Example input with 30 features (must match training dataset!)
    sample_input = np.array([[
        17.99, 10.38, 122.8, 1001.0, 0.1184, 
        0.2776, 0.3001, 0.1471, 0.2419, 0.0787,
        1.095, 0.9053, 8.589, 153.4, 0.0064, 
        0.049, 0.0537, 0.015, 0.0305, 0.0062,
        25.38, 17.33, 184.6, 2019.0, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]])

    # Make prediction
    prediction = model.predict(sample_input)
    print("Prediction (0=benign, 1=malignant):", prediction)

if __name__ == "__main__":
    predict_logistic()
