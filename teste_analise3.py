import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import numpy as np
import seaborn as sns
import scipy.integrate as sp
import pickle


with open("df.pkl", "rb") as f:
    df = pickle.load(f)


fig, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax) 
ax.set_title("Matriz de Correlação df") 

X = pd.to_datetime(df["Timestamp"], unit="s")
Y = df['Close']
#derivadas
# Y_diff = np.diff(df['Close'])
Y_grad = np.gradient(df['Close'])
# Y_diff_abs = np.absolute(Y_diff)
Y_grad_abs = np.absolute(Y_grad)

#integral
Y_cum = np.cumsum(df['Close'])

#outros
# Y_grad_fft = np.fft.rfft(Y_grad)
# Y_grad_ifft = np.fft.irfft(Y_grad)

K = [1, 10, 50, 100, 150, 365]
K2 = [50, 365]
with open("hip.pkl", "rb") as f:
    hip = pickle.load(f)

fig, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(hip.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax) 
ax.set_title("Matriz de Correlação hip") 

with open("alphas.pkl", "rb") as f:
    alphas = pickle.load(f)

fig, axes = plt.subplots(len(K2), 1, figsize=(10, 5 * len(K2)))

for ax, k in zip(axes, K2):
    ax.plot(X, alphas[f"alpha_{k}"], color="tab:blue")
    ax.set_title(f"Alpha_{k}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X, Y, color="tab:blue", label="")
ax.set_title(f"Bitcoin (Close)") 
ax.set_xlabel("Data") 
ax.set_ylabel("Valor")

fig, bgx = plt.subplots(figsize=(10, 5))
bgx.plot(X, Y_grad, color="tab:blue", label="")
bgx.set_title(f"Bitcoin grad (Close)") 
bgx.set_xlabel("Data") 
bgx.set_ylabel("Valor")

fig, cx = plt.subplots(figsize=(10, 5))
cx.plot(X, hip["Y_cum_50"], color="tab:blue", label="")
cx.set_title(f"Bitcoin integral 50 (Close)") 
cx.set_xlabel("Data") 
cx.set_ylabel("Valor")

plt.tight_layout()

# for i, y in enumerate(Y_grad):
#     if y == 0:
#         print(y, X[i])

# print("PRINT")

# for i, y in enumerate(hip["Y_cum_50"]):
#     if y == 0:
#         print(y, X[i])
plt.show()