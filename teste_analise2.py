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

with open("df.pkl", "wb") as f:
    pickle.dump(df, f)

print("SALVO DF")


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


dia = 1440
K = [1, 10, 50, 100, 150, 365]
dicionario = {"Y_grad": Y_grad, "Y_grad_abs": Y_grad_abs}

for k in K:
    temp = [((Y_cum[i] - Y_cum[i - k*dia]) if i >= k*dia else Y_cum[i]) for i in range(len(Y_cum))]
    dicionario[f"Y_cum_{k}"] = temp

with open("dicionario.pkl", "wb") as f:
    pickle.dump(dicionario, f)

print("SALVO dicionario")

hip = pd.DataFrame(dicionario)

with open("hip.pkl", "wb") as f:
    pickle.dump(hip, f)

print("SALVO hip")

fig, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(hip.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax) 
ax.set_title("Matriz de Correlação hip") 

alphas = {}
K2 = [50, 365]
for k in K2:
    alpha = [hip[f"Y_cum_{k}"][i] / hip["Y_grad"][i] if hip["Y_grad"][i] != 0 else np.nan for i in range(len(hip[f"Y_cum_{k}"]))]
    alphas[f"alpha_{k}"] = alpha

with open("alphas.pkl", "wb") as f:
    pickle.dump(alphas, f)

print("SALVO alphas")

fig, axes = plt.subplots(len(K), 1, figsize=(10, 5 * len(K)))

for ax, k in zip(axes, K2):
    ax.plot(X, alphas[f"alpha_{k}"], color="tab:blue")
    ax.set_title(f"Alpha_{k}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor")

plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(X, Y, color="tab:blue", label="")
# ax.set_title(f"Bitcoin (Close)") 
# ax.set_xlabel("Data") 
# ax.set_ylabel("Valor")

# fig, bx = plt.subplots(figsize=(10, 5))
# bx.plot(X[:-1], Y_diff, color="tab:blue", label="")
# bx.set_title(f"Bitcoin diff (Close)") 
# bx.set_xlabel("Data") 
# bx.set_ylabel("Valor")

# fig, bgx = plt.subplots(figsize=(10, 5))
# bgx.plot(X, Y_grad, color="tab:blue", label="")
# bgx.set_title(f"Bitcoin grad (Close)") 
# bgx.set_xlabel("Data") 
# bgx.set_ylabel("Valor")

# fig, cx = plt.subplots(figsize=(10, 5))
# cx.plot(X, Y_cum, color="tab:blue", label="")
# cx.set_title(f"Bitcoin integral (Close)") 
# cx.set_xlabel("Data") 
# cx.set_ylabel("Valor")

# fig, dx = plt.subplots(figsize=(10, 5))
# dx.plot(X[:-1], Y_diff_abs, color="tab:blue", label="")
# dx.set_title(f"Bitcoin diff abs (Close)") 
# dx.set_xlabel("Data") 
# dx.set_ylabel("Valor")

# fig, dgx = plt.subplots(figsize=(10, 5))
# dgx.plot(X, Y_grad_abs, color="tab:blue", label="")
# dgx.set_title(f"Bitcoin grad abs (Close)") 
# dgx.set_xlabel("Data") 
# dgx.set_ylabel("Valor")

# fig, ex = plt.subplots(figsize=(10, 5))
# ex.plot([i for i in range(len(Y_grad_fft))], Y_grad_fft, color="tab:blue", label="")
# ex.set_title(f"Bitcoin  grad fft (Close)") 
# ex.set_xlabel("Data") 
# ex.set_ylabel("Valor")

# fig, fx = plt.subplots(figsize=(10, 5))
# fx.plot([i for i in range(len(Y_grad_ifft))], Y_grad_ifft, color="tab:blue", label="")
# fx.set_title(f"Bitcoin  grad fft (Close)") 
# fx.set_xlabel("Data") 
# fx.set_ylabel("Valor")

plt.show()