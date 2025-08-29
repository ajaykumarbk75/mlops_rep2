# Save as logistic_classifier_mlflow.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

def train_logistic_model():
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        # Load dataset
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)

        # Signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log and register
        mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic-reg-model",
            input_example=X_train[:5],
            signature=signature,
            registered_model_name="logistic-classifier-model"
        )

        print("✅ Logistic Regression model registered successfully.")

if __name__ == "__main__":
    train_logistic_model()
