import joblib
import pandas as pd
import numpy as np

model = joblib.load("Models\\linear_reg_model.pkl")

new_data = pd.DataFrame([[123, 56, 89]])

prediction = model.predict(new_data)

print("Predicted Sales:", prediction) 