import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose

from core.dataset_segregator import fetch_dataset_as_json

def timestamp_indexing(data):
    for i, dt in enumerate(data):
        dt = dt.reset_index()
        dt['point_timestamp'] = pd.to_datetime(dt['point_timestamp'])
        dt = dt.set_index('point_timestamp')
        dt = dt[~dt.index.duplicated(keep='first')]
        data[i] = dt
    return data

def frequency_setter(data, features, i):
    val = features[i]['frequency']
    if val == 'hourly':
        data[i] = data[i].resample('h').interpolate(method='linear')
        period = 24
    elif val == 'daily':
        data[i] = data[i].resample('D').interpolate(method='linear')
        period = 7
    elif val == 'monthly':
        data[i] = data[i].resample('ME').mean().interpolate(method='linear')
        if data[i].dropna().empty:
            data[i] = data[i].resample('MS').mean().interpolate(method='linear')
        period = 12
    elif val == 'weekly':
        data[i] = data[i].resample('W-MON').interpolate(method='linear')
        period = min(52, len(data[i]) // 2)
    else:
        data[i] = data[i].resample('Y').interpolate(method='linear')
        period = 1
    return data[i], period

def adf_and_seasonal(data, features):
    data = timestamp_indexing(data)

    addon_features = []
    valid_data = []
    valid_features = []

    for i in range(len(data)):
        data[i], period = frequency_setter(data, features, i)
        data[i] = data[i].dropna(subset=['point_value'])

        if not data[i].empty:
            series = data[i]['point_value']

            if series.nunique() <= 1:
                # print(f"Skipping feature {i}, constant series")
                continue

            if len(series) < 2 * period:
                # print(f"Skipping feature {i}, not enough data for seasonal decomposition (has {len(series)}, needs at least {2*period})")
                continue

            try:
                adf = adfuller(series)
                seasonal = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
                seasonal_component = seasonal.seasonal
                trend = seasonal.trend
                residual_component = seasonal.resid
                seasonality_strength = seasonal_component.var() / features[i]['var']
                seasonal_presence = seasonal_component.abs().mean() > (residual_component.abs().mean() * 0.5)
                acf1 = acf(series, 1, fft=True, alpha=0.05)

                addon_features.append({
                    "seasonal": bool(seasonal_presence),
                    "seasonality_strength": seasonality_strength,
                    "trend": trend.mean(),
                    "period": period,
                    "adf_stat": float(adf[0]),
                    "adf_pvalue": float(adf[1]),
                    "acf1": acf1[0][1]
                })

                valid_data.append(data[i])
                valid_features.append(features[i])

            except Exception as e:
                print(f"Error processing feature {i}: {e}")
                continue

    return valid_data, valid_features, addon_features

# Example usage
# features, data = fetch_dataset_as_json()
# dataset, features_set, addon_feature_set = adf_and_seasonal(data, features)
# print(features_set)
# print(addon_feature_set)
