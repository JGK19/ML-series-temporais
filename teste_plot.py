import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import numpy as np

path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
dataset_path = os.path.join(path, "btcusd_1-min_data.csv")

df = pd.read_csv(dataset_path)

time_to_data = np.vectorize(datetime.fromtimestamp)
X = time_to_data(df['Timestamp'])


series = ['Open','High','Low','Close','Volume']
ax = [None for _ in series]

for i, serie in enumerate(series):
    Y = df[serie]
    fig, ax[i] = plt.subplots(figsize=(10, 5))
    ax[i].plot(X, Y, color="tab:blue", label="")

    ax[i].set_title(f"Bitcoin ({serie})") 
    ax[i].set_xlabel("Data") 
    ax[i].set_ylabel("Valor")

plt.show()

print(df)