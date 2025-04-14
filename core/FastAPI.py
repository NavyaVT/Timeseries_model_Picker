import pandas as pd
import numpy as np
import joblib
from scipy.stats import entropy
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from dataset_segregator import freq_detecter
from make_predictions import gru_model, fourier_model, prophet_model, ets_model

app = FastAPI()

classifier, label_enc = joblib.load('D:\\DataGenie_DS\\Datagenie_DS\\models\\random_forest_model.pkl')

def timestamp_indexing(dt):
    dt['point_timestamp'] = pd.to_datetime(dt['point_timestamp'])
    dt = dt.set_index('point_timestamp')
    dt = dt[~dt.index.duplicated(keep='first')]
    return dt

def extract_features(data):
    data = timestamp_indexing(data)
    freq = freq_detecter(data)
    if freq == 'hourly':
        data = data.resample('h').interpolate(method='linear')
        period = 24
    elif freq == 'daily':
        data = data.resample('D').interpolate(method='linear')
        period = 7
    elif freq == 'monthly':
        data = data.resample('ME').mean().interpolate(method='linear')
        if data.dropna().empty:
            data = data.resample('MS').mean().interpolate(method='linear')
        period = 12
    elif freq == 'weekly':
        data = data.resample('W-MON').interpolate(method='linear')
        period = min(52, len(data) // 2)
    else:
        data = data.resample('YE').interpolate(method='linear')
        period = 1
    series = data['point_value']
    adf = adfuller(series)
    if(len(series)>period*2):
        seasonal = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        seasonal_component = seasonal.seasonal
        trend = seasonal.trend
        seasonality_strength = seasonal_component.var() / data['point_value'].var()
    else:
        seasonality_strength = 0
        trend = 0
    acf1 = acf(series, 1, fft=True, alpha=0.05)
    features = {
        "seasonality_strength": seasonality_strength,
        "trend": trend.mean(),
        "period": period,
        "adf_pvalue": float(adf[1]),
        "acf1": acf1[0][1],
        "std_deviation": float(data['point_value'].std()),
        "skew": float(data['point_value'].skew()),
        "kurtosis": float(data['point_value'].kurtosis()),
        "entropy": float(entropy(data['point_value'].abs())),
    }
    return features

def detect_anomalies(actual, predicted, threshold=90):
    residuals = np.abs(actual - predicted)
    threshold_value = threshold * np.std(residuals)
    return ["yes" if r > threshold_value else "no" for r in residuals]

def calculate_forecastability_score(data, features):
    score = 0
    if 'std_deviation' in features and data.mean() != 0:
        cv = features['std_deviation'] / data.mean()
        score += 2.0 * min(1.0, 1.0/max(cv, 0.001))
    if 'trend' in features:
        trend_norm = min(1.0, abs(features['trend']) / max(features['std_deviation'], 0.001))
        score += 3.0 * (1.0 - trend_norm)
    if 'seasonality_strength' in features:
        score += 3.0 * min(1.0, max(0, features['seasonality_strength']))
    randomness = 0
    if 'skew' in features:
        randomness += 0.5 * min(1.0, abs(features['skew']))
    if 'kurtosis' in features:
        randomness += 0.5 * min(1.0, max(0, features['kurtosis'])/10)
    score += 2.0 * (1.0 - min(1.0, randomness))
    return min(10.0, max(0.0, round(score, 1)))

class TestData(BaseModel):
    timestamp: str
    point_value: float

class PredictionRequest(BaseModel):
    test_data: List[TestData]
    date_from: str
    date_to: str

@app.post("/predict")
async def predict_timeseries_endpoint(
    file: UploadFile = File(...),
    request_data: str = Form(...)
):
    try:
        request = PredictionRequest.parse_raw(request_data)
        file_path = file.file

        file_ext = file.filename.split('.')[-1].lower()
        if file_ext == 'csv':
            data = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            data = pd.read_excel(file_path)
        elif file_ext == 'txt':
            data = pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError("Unsupported file format")

        test_data_list = [item.dict() for item in request.test_data]

        result = predict_timeseries_standalone(data, test_data_list, request.date_from, request.date_to)
        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def predict_timeseries_standalone(data, test_data, date_from, date_to):
    try:
        test = pd.DataFrame(test_data)
        test['point_timestamp'] = pd.to_datetime(test['timestamp'])
        test = test[['point_timestamp', 'point_value']]

        features = extract_features(data)
        features_array = np.array([list(features.values())])
        best_model_name = label_enc.inverse_transform(classifier.predict(features_array))
        best_model_name = 'ets'
        train_data = data[data['point_timestamp'] < pd.to_datetime(date_from)]
        train_data = train_data[['point_timestamp', 'point_value']]
        test_for_model = test.copy()

        start_time = time.time()
        if best_model_name == 'ets':
            mape, prediction = ets_model(train_data, test_for_model, features)
        elif best_model_name == 'gru':
            mape, prediction = gru_model(train_data, test_for_model)
        elif best_model_name == 'prophet':
            mape, prediction = prophet_model(train_data, test_for_model)
        elif best_model_name == 'fourier':
            mape, prediction = fourier_model(train_data, test_for_model, features)

        processing_time = time.time() - start_time
        anomalies = detect_anomalies(test['point_value'].values, prediction)
        forecastability_score = calculate_forecastability_score(train_data['point_value'], features)

        results = []
        for i in range(len(test)):
            results.append({
                "timestamp": test.iloc[i]['point_timestamp'].isoformat(),
                "point_value": float(test.iloc[i]['point_value']),
                "predicted": float(prediction[i]),
                "is_anomaly": anomalies[i]
            })

        return {
            "forecastability_score": round(float(forecastability_score), 1),
            "number_of_batch_fits": len(train_data) // 100 + 1,
            "mape": float(mape),
            "avg_time_taken_per_fit_in_seconds": round(processing_time, 2),
            "total_processing_time_seconds": round(processing_time, 2),
            "best_model": best_model_name[0] if isinstance(best_model_name, np.ndarray) else best_model_name,
            "results": results
        }

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")