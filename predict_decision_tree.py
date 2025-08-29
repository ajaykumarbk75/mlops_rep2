import mlflow
import mlflow.sklearn

def predict_decision_tree():
    mlflow.set_tracking_uri("http://localhost:5000")

    model = mlflow.sklearn.load_model("models:/decision-tree-regressor-model/1")

    # Example input: 10 features matching diabetes dataset
    sample_input = [[0.038075906, 0.05068012, 0.061696208, 0.021872355,
                     -0.044223498, -0.034820765, -0.043400846, -0.002592262,
                     0.01990749, -0.017646125]]

    prediction = model.predict(sample_input)
    print("Prediction (regression output):", prediction)

if __name__ == "__main__":
    predict_decision_tree()
