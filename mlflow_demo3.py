# Save as decision_tree_regressor_mlflow.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

def train_decision_tree_model():
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        # Load dataset
        data = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        # Train model
        model = DecisionTreeRegressor(max_depth=4)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("mse", mse)

        # Signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log and register
        mlflow.sklearn.log_model(
            sk_model=model,
            name="dt-reg-model",
            input_example=X_train[:5],
            signature=signature,
            registered_model_name="decision-tree-regressor-model"
        )

        print("✅ Decision Tree Regressor model registered successfully.")

if __name__ == "__main__":
    train_decision_tree_model()
