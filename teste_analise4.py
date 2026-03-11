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


path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
dataset_path = os.path.join(path, "btcusd_1-min_data.csv")

df = pd.read_csv(dataset_path)

# with open("df.pkl", "wb") as f:
#     pickle.dump(df, f)

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

hip = pd.DataFrame(dicionario)

# with open("hip.pkl", "wb") as f:
#     pickle.dump(hip, f)

print("SALVO hip")

fig, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(hip.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax) 
ax.set_title("Matriz de Correlação hip") 

alphas = {}
K2 = [50, 365]
for k in K2:
    alpha = [hip[f"Y_cum_{k}"][i] / hip["Y_grad_abs"][i] if hip["Y_grad_abs"][i] != 0 else np.nan for i in range(len(hip[f"Y_cum_{k}"]))]
    alphas[f"alpha_{k}"] = alpha

# with open("alphas.pkl", "wb") as f:
#     pickle.dump(alphas, f)

print("SALVO alphas")

fig, axes = plt.subplots(len(K), 1, figsize=(10, 5 * len(K)))

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

fig, dx = plt.subplots(figsize=(10,5))
data = np.array(alphas["alpha_50"])
data = data[~np.isnan(data)]
dx.boxplot(data)
dx.set_title("Distribuição de alpha_50")
dx.set_ylabel("alpha_50")


fig, ex = plt.subplots(figsize=(10,5))
data = np.array(alphas["alpha_50"])
data = data[~np.isnan(data)]
ex.hist(data, bins=100)
ex.set_title("Histograma alpha_50")
ex.set_xlabel("alpha")
ex.set_ylabel("frequência")


fig, fx = plt.subplots(figsize=(10,5))
data = np.array(alphas["alpha_50"])
data = data[~np.isnan(data)]
fx.hist(data, bins=100)
fx.set_title("Histograma alpha_50 log")
fx.set_xlabel("alpha")
fx.set_ylabel("frequência")
fx.set_xscale("log")

fig, gx = plt.subplots(figsize=(10,5))
sns.kdeplot(data, fill=True)
gx.set_title("Distribuição KDE alpha_50")

fig, hx = plt.subplots(figsize=(10,5))

sorted_data = np.sort(data)
p = np.arange(len(data)) / len(data)
hx.plot(sorted_data, p)
hx.set_title("ECDF alpha_50")
hx.set_xlabel("alpha")
hx.set_ylabel("probabilidade")

fig, ix = plt.subplots(figsize=(6,6))
stats.probplot(data, dist="norm", plot=ix)

fig, jx = plt.subplots(figsize=(6,6))
jx.violinplot(data)
jx.set_title("Violin plot alpha_50")

plt.tight_layout()
plt.show()