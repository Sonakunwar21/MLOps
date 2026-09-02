import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib


df = pd.read_csv("C:\\Users\\saura\\chapter 1\\MLOps\\data\\Data.csv", index_col=0)
df.head()

#Train-Test split

X,y = df[["TV", "radio", "newspaper"]], df[["sales"]]

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.2, random_state = 64)


model = LinearRegression()
model.fit(Xtrain , ytrain)

ypred = model.predict(Xtest)
rmse = root_mean_squared_error(ytest,ypred)
r2 = r2_score(ytest,ypred)

print(f"RMSE:{rmse}")
print(f"r2:{r2}")

# Model dump
joblib.dump(model,"Models\\linear_reg_model.pkl")
