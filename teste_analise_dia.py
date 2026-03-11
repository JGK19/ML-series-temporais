import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
import scipy.integrate as sp

# download dataset
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

print(len(df_daily))
# remover dias sem negociação
df_daily = df_daily.dropna()
print(len(df_daily))

# usar dataset diário
df = df_daily.reset_index()

# -----------------------------
# CORRELAÇÃO
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Matriz de Correlação df")

# -----------------------------
# SERIES
# -----------------------------
X = df["datetime"]
Y = df["Close"]

# -----------------------------
# DERIVADAS
# -----------------------------
Y_grad = np.gradient(Y)
Y_grad_abs = np.abs(Y_grad)

# -----------------------------
# INTEGRAL
# -----------------------------
Y_cum = np.cumsum(Y)

# -----------------------------
# GRAFICO PREÇO
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X, Y, color="tab:blue")
ax.set_title("Bitcoin (Close) - Daily")
ax.set_xlabel("Data")
ax.set_ylabel("Valor")

# -----------------------------
# GRAFICO DERIVADA
# -----------------------------
fig, bgx = plt.subplots(figsize=(10, 5))
bgx.plot(X, Y_grad, color="tab:blue")
bgx.set_title("Bitcoin grad (Close) - Daily")
bgx.set_xlabel("Data")
bgx.set_ylabel("Valor")

plt.show()