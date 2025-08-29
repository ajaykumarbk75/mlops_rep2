import mlflow.sklearn

# Load the model by name and version
model = mlflow.sklearn.load_model("models:/sklearn-rf-reg-model/1")

# Example input for prediction
predictions = model.predict([[0.1, -0.2, 0.3, 0.4]])

# Output the predictions
print("Sample predictions:", predictions)
