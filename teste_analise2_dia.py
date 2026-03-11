import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import numpy as np
import seaborn as sns
import scipy.integrate as sp
import pickle

path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
dataset_path = os.path.join(path, "btcusd_1-min_data.csv")

df = pd.read_csv(dataset_path)

# -----------------------------
# CONVERTER TIMESTAMP
# -----------------------------
df["datetime"] = pd.to_datetime(df["Timestamp"], unit="s")
df = df.set_index("datetime")

# -----------------------------
# RESAMPLE PARA 1 DIA
# -----------------------------
df_daily = pd.DataFrame()

df_daily["Open"] = df["Open"].resample("1D").first()
df_daily["Close"] = df["Close"].resample("1D").last()
df_daily["High"] = df["High"].resample("1D").max()
df_daily["Low"] = df["Low"].resample("1D").min()
df_daily["Volume"] = df["Volume"].resample("1D").sum()

df_daily = df_daily.dropna()

# SUBSTITUI DATASET ORIGINAL
df = df_daily.reset_index()

with open("df_daily.pkl", "wb") as f:
    pickle.dump(df, f)

print("SALVO DF")

# -----------------------------
# CORRELAÇÃO
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax) 
ax.set_title("Matriz de Correlação df_dia") 

X = df["datetime"]
Y = df["Close"]

# -----------------------------
# DERIVADAS
# -----------------------------
Y_grad = np.gradient(df["Close"])
Y_grad_abs = np.absolute(Y_grad)

# -----------------------------
# INTEGRAL
# -----------------------------
Y_cum = np.cumsum(df["Close"])

# -----------------------------
# JANELAS
# -----------------------------
K = [1, 10, 50, 100, 150, 365]

dicionario = {"Y_grad": Y_grad, "Y_grad_abs": Y_grad_abs}

for k in K:
    temp = [
        (Y_cum[i] - Y_cum[i-k]) if i >= k else Y_cum[i]
        for i in range(len(Y_cum))
    ]
    dicionario[f"Y_cum_{k}"] = temp

hip = pd.DataFrame(dicionario)

with open("hip_daily.pkl", "wb") as f:
    pickle.dump(hip, f)

print("SALVO hip")

# -----------------------------
# CORRELAÇÃO HIP
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(hip.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax) 
ax.set_title("Matriz de Correlação hip") 

# -----------------------------
# ALPHAS
# -----------------------------
alphas = {}
K2 = [50, 365]

for k in K2:
    alpha = [
        hip[f"Y_cum_{k}"][i] / hip["Y_grad"][i]
        if hip["Y_grad"][i] != 0
        else np.nan
        for i in range(len(hip))
    ]
    alphas[f"alpha_{k}"] = alpha

with open("alphas_daily.pkl", "wb") as f:
    pickle.dump(alphas, f)

print("SALVO alphas")

# -----------------------------
# GRAFICOS
# -----------------------------
fig, axes = plt.subplots(len(K2), 1, figsize=(10, 5 * len(K2)))

for ax, k in zip(axes, K2):
    ax.plot(X, alphas[f"alpha_{k}"], color="tab:blue")
    ax.set_title(f"Alpha_{k}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor")

plt.tight_layout()
plt.show()