import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import numpy as np
import seaborn as sns
import scipy.integrate as sp
import pickle
import scipy.stats as stats

#classificação
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

#regressão
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.model_selection import train_test_split

with open("hip.pkl", "rb") as f:
    hip = pickle.load(f)

print(hip)
#df = hip[["Y_grad_abs", "Y_cum_50"]][:-1]
df = hip.drop("Y_grad", axis=1)

X = df.drop("Y_grad_abs", axis=1)
y = df["Y_grad_abs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

reg_linear = LinearRegression()
reg_linear.fit(X_train, y_train)
#y_pred_linear = reg_linear.predict(X_test)
print(f"linear: {reg_linear.score(X_test, y_test)}")

reg_knn = KNeighborsRegressor(n_neighbors=5)
reg_knn.fit(X_train, y_train)
#y_pred_knn = reg_knn.predict(X_test)
print(f"knn: {reg_knn.score(X_test, y_test)}")

reg_gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
reg_gbr.fit(X_train, y_train)
#y_pred_gbr = reg_gbr.predict(X_test)
print(f"gbr: {reg_gbr.score(X_test, y_test)}")
