import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import entropy
import os

#formatting setting
pd.options.display.float_format = '{:,.2f}'.format

# extracting dataset and segregating into json file with time frequency
def freq_detecter(data):
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    freq = pd.infer_freq(data.index)
    if freq is None:
        deltas = data.index.to_series().diff().dropna()
        deltas = deltas[deltas > pd.Timedelta(0)]

        if deltas.empty:
            return "unknown"
        most_common_delta = deltas.mode()[0]

        if most_common_delta == pd.Timedelta(days=1):
            return "daily"
        elif most_common_delta == pd.Timedelta(weeks=1):
            return "weekly"
        elif most_common_delta <= pd.Timedelta(hours=1):
            return "hourly"
        elif (most_common_delta <= pd.Timedelta(days=30) or most_common_delta <= pd.Timedelta(days=31)):
            return "monthly"
    else:
        freq_map = {'D': 'daily', 'W': 'weekly', 'h': 'hourly', 'MS': 'monthly', 'ME': 'monthly'}
        return freq_map.get(freq, freq)

def fetch_dataset_as_json():
    dataset = []
    values = []
    paths = ["daily","hourly","monthly","weekly","new"]
    # paths = ["new"]
    for path in paths:
        for i in range(1,41):
            file_path = os.path.join("datasets",path, f"sample_{i}.csv")
            if os.path.exists(file_path): #check if file exists.
                data = pd.read_csv(file_path,header=0, index_col=0)
                data.dropna(inplace=True)
                data.drop_duplicates(inplace=True)
                values.append(data)
                if(path=='new'):
                    freq = freq_detecter(data)
                else: freq = path
                dataset.append({"frequency": freq,
                                "mean": float(data['point_value'].mean()),
                                "std_deviation": float(data['point_value'].std()),
                                "var":float(data['point_value'].var()),
                                "skew": float(data['point_value'].skew()),
                                "kurtosis": float(data['point_value'].kurtosis()),
                                "entropy": float(entropy(data['point_value'].abs())),
                                })
                # print(data)
                # print(freq)
    return values, dataset

# val, ds = fetch_dataset_as_json()
# print(ds)