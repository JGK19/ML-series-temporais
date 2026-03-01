import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import numpy as np
import seaborn as sns
import scipy.integrate as sp

path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
dataset_path = os.path.join(path, "btcusd_1-min_data.csv")

df = pd.read_csv(dataset_path)



fig, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax) 
ax.set_title("Matriz de Correlação (Seaborn)") 

time_to_data = np.vectorize(datetime.fromtimestamp)
X = time_to_data(df['Timestamp'])
Y = df['Close']
Y_diff = np.diff(df['Close'])
Y_grad = np.gradient(df['Close'])
Y_cum = np.cumsum(df['Close'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X, Y, color="tab:blue", label="")
ax.set_title(f"Bitcoin (Close)") 
ax.set_xlabel("Data") 
ax.set_ylabel("Valor")

fig, bx = plt.subplots(figsize=(10, 5))
bx.plot(X, Y_diff, color="tab:blue", label="")
bx.set_title(f"Bitcoin diff (Close)") 
bx.set_xlabel("Data") 
bx.set_ylabel("Valor")

fig, bgx = plt.subplots(figsize=(10, 5))
bgx.plot(X, Y_grad, color="tab:blue", label="")
bgx.set_title(f"Bitcoin grad (Close)") 
bgx.set_xlabel("Data") 
bgx.set_ylabel("Valor")

fig, cx = plt.subplots(figsize=(10, 5))
cx.plot(X, Y_cum, color="tab:blue", label="")
cx.set_title(f"Bitcoin integral (Close)") 
cx.set_xlabel("Data") 
cx.set_ylabel("Valor")

plt.show()