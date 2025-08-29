# Install dependencies if not already done
# pip install mlflow scikit-learn

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

def train_log_and_register():
    # Start a new MLflow run
    with mlflow.start_run() as run:
        # Generate synthetic regression data
        X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple RandomForestRegressor
        params = {"max_depth": 2, "random_state": 42}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Log hyperparameters
        mlflow.log_params(params)

        # Predict and log the MSE
        y_pred = model.predict(X_test)
        mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

        # Infer model signature (input–output schema)
        signature = infer_signature(X_train, model.predict(X_train))

        # Log and register the model in one step
        mlflow.sklearn.log_model(
            sk_model=model,
            name="sklearn-regressor",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="sklearn-rf-reg-model"
        )

        print("Model training, logging, and registration completed.")

if __name__ == "__main__":
    train_log_and_register()
