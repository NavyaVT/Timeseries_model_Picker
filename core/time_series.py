import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from pmdarima import auto_arima
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.deterministic import DeterministicProcess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

from core.feature_selection import adf_and_seasonal
from core.dataset_segregator import fetch_dataset_as_json

import warnings
import logging
import os
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None
logger = logging.getLogger("cmdstanpy")
logger.setLevel(logging.CRITICAL)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(logging.NullHandler())

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def arima_model(data, addon_features):
    train_size = int(len(data) * 0.8)
    train, test = data['point_value'][:train_size], data['point_value'][train_size:]
    try:
        model = auto_arima(train, seasonal=addon_features['seasonal'], m=addon_features['period'])
        predictions = model.predict(n_periods=len(test))
        return [calculate_mape(test, predictions), predictions]
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return float('inf')

def ets_features(features, addon):
    if(features['var'] > features['mean'] or features['kurtosis'] > 2.5):
        error = 'add'
    else:
        error = 'add'
    if(addon['adf_pvalue'] < 0.05 or (features['skew'] < 0.1)):
        trend = None
    elif(addon['adf_pvalue'] >= 0.05 or (features['skew'] < 0 and features['skew'] < 1)):
        trend = 'add'
    else:
        trend = 'add'
    if(addon['seasonal'] == False):
        season = None
    if(features['entropy'] < 4 or addon['adf_pvalue'] < 0.05):
        season = None
    elif(features['entropy'] >= 4 and features['entropy'] <= 6):
        season = 'add'
    else:
        season = 'add'
    return error, trend, season

def ets_model(data, features,addon_features):
    # print("data lenght = ",len(data),"\nperiod = ",addon_features['period'])
    train_size = int(len(data) * 0.8)
    train, test = data['point_value'][:train_size], data['point_value'][train_size:]
    err, trend, season = ets_features(features,addon_features)
    if(len(train)< (2*addon_features['period'])):
        period = None
        season = None
    else: period = addon_features['period']
    # print(data, train, test)
    try:
        model = ETSModel(train, error=err, trend=trend, seasonal=season, seasonal_periods=period).fit()
        predictions = model.forecast(steps=len(test))
        return [calculate_mape(test, predictions), predictions]
    except Exception as e:
        print(f"ETS Error: {e}")
        return float('inf')

def prophet_model(data):
    dt = data.copy()
    dt = data.reset_index()
    dt = dt.rename(columns={'point_timestamp': 'ds', 'point_value': 'y'})

    train_size = int(len(dt) * 0.8)
    train, test = dt[:train_size], dt[train_size:]
    
    train.loc[:, 'ds'] = pd.to_datetime(train['ds'])
    test.loc[:, 'ds'] = pd.to_datetime(test['ds'])
    
    try:
        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        predictions = forecast['yhat'][train_size:].values
        return [calculate_mape(test['y'], predictions), predictions]
    except Exception as e:
        print(f"Prophet Error: {e}")
        return float('inf')
    
def get_fourier_order(series_length,addon_features):
    if addon_features['period'] <= 7:
        if addon_features['seasonality_strength'] > 0.7 and addon_features['acf1'] > 0.6:
            return 3
        elif addon_features['acf1'] < 0.3:
            return 1
        else:
            return 2
    elif addon_features['period'] <= 30:
        if series_length > 200 and addon_features['seasonality_strength'] > 0.6:
            return 4
        else:
            return 2
    elif addon_features['period'] > 30:
        if addon_features['acf1'] > 0.7:
            return 5
        else:
            return 3
    return 2

def fourier_model(data, addon_features):
    y = data['point_value'].values
    t = data.index

    train_size = int(len(data) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    t_train, t_test = t[:train_size], t[train_size:]

    try:
        fourier_order = get_fourier_order(len(data),addon_features)
        fourier = Fourier(period=addon_features['period'], order=fourier_order)

        dp_full = DeterministicProcess(index=t, constant=False, order=0, seasonal=addon_features['seasonal'], additional_terms=[fourier])
        X_full = dp_full.in_sample()

        X_train, X_test = X_full[:train_size], X_full[train_size:]

        model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return [calculate_mape(y_test, predictions), predictions]
    except Exception as e:
        print(f"Fourier MLPRegressor Error: {e}")
        return float('inf')

def gru_model(data):
    y = data['point_value'].values
    if len(y) < 1:
        return float('inf')
    y_min, y_max = y.min(), y.max()
    if y_max == y_min:
        return 0.0 if len(y) == len(set(y)) else float('inf')
    y_norm = (y - y_min) / (y_max - y_min)
    train_size = max(10, int(len(y) * 0.8))
    y_train, y_test = y_norm[:train_size], y_norm[train_size:]
    seq_len = min(3, len(y_train) // 5)
    if len(y_train) <= seq_len or len(y_test) <= seq_len:
        return float('inf')

    def create_sequences(data):
        return np.lib.stride_tricks.sliding_window_view(data, seq_len)[:-1], data[seq_len:]

    X_train, y_train_seq = create_sequences(y_train)
    X_test, y_test_seq = create_sequences(y_test)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(1, 8, batch_first=True)
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(8, 1)

        def forward(self, x):
            x, _ = self.gru(x)
            return self.fc(self.dropout(x[:, -1, :])).squeeze()

    model = GRUModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    best_mape = float('inf')

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = F.mse_loss(outputs, torch.tensor(y_train_seq, dtype=torch.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_test).clamp(0, 1)
            mape = calculate_mape(y_test_seq, preds.numpy())
            scheduler.step(mape)
        if mape < best_mape:
            best_mape = mape
            if best_mape < 1.0:
                break
    with torch.no_grad():
        final_preds = model(X_test).numpy() * (y_max - y_min) + y_min
    return [calculate_mape(y_test_seq * (y_max - y_min) + y_min, final_preds), final_preds]
  
d,f = fetch_dataset_as_json()
#print(d.columns)

dataset, features_set, addon_feature_set = adf_and_seasonal(d,f)

# for i in range(0, len(dataset)):
#     print(addon_feature_set[i]['seasonality_strength'])
models = []
for i in range(0, len(dataset)):
    # print("Lenght of data is ", len(dataset))
    mape_values = {"arima" : arima_model(data=dataset[i],addon_features=addon_feature_set[i]),
                   "ets" : ets_model(data=dataset[i],features=features_set[i],addon_features=addon_feature_set[i]),
                   "prophet" : prophet_model(data=dataset[i]),
                   "fourier" : fourier_model(data=dataset[i],addon_features=addon_feature_set[i]),
                   "GRU" : gru_model(data=dataset[i])}
    # print(mape_values)
    best_model = min(mape_values, key=mape_values.get)
    print(f"Best model: {best_model}, Best MAPE: {mape_values[best_model]}")
    models.append(best_model)

labels = ["seasonality_strength", "trend","period","adf_pvalue",
                   "acf1","std_deviation","skew","kurtosis","entropy","model" ]

data_with_label=pd.DataFrame(columns=labels)
for i in range(0,len(dataset)):
    data_with_label.loc[i] = [addon_feature_set[i]['seasonality_strength'],
                         addon_feature_set[i]['trend'],
                         addon_feature_set[i]['period'],
                         addon_feature_set[i]['adf_pvalue'],
                         addon_feature_set[i]['acf1'],
                         features_set[i]['std_deviation'],
                         features_set[i]['skew'],
                         features_set[i]['kurtosis'],
                         features_set[i]['entropy'],
                         models[i]]
    
print(data_with_label)
data_with_label.to_csv('data_with_labels.csv',index=False)


