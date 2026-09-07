import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load saved model
model = mlflow.sklearn.load_model(
    "models:/Sales_Prediction_Model@champion"
)
# New observation
new_data = pd.DataFrame({"TV": [35],"radio": [50000],"newspaper": [8]})
# Prediction
prediction = model.predict(new_data)
print("Prediction:", prediction[0]) 