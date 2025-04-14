from fastapi import FastAPI, File, Form, UploadFile, Query, Body, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, List, Union
import joblib
import io
import os
import tempfile
from scipy.stats import entropy
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import time
import json

from core.dataset_segregator import freq_detecter
from Datagenie_DS.core.make_predictions import gru_model, fourier_model, prophet_model, ets_model ,arima_model

classifier, label_enc = joblib.load('models\\random_forest_model.pkl')
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
    features = {"seasonality_strength": seasonality_strength,
                "trend":trend.mean(),
                "period":period,
                "adf_pvalue":float(adf[1]),
                "acf1": acf1[0][1],
                "std_deviation": float(data['point_value'].std()),
                "skew": float(data['point_value'].skew()),
                "kurtosis": float(data['point_value'].kurtosis()),
                "entropy": float(entropy(data['point_value'].abs())),
                }
    return features

def detect_anomalies(actual, predicted, threshold=2.0):
    residuals = np.abs(actual - predicted)
    threshold_value = threshold * np.std(residuals)
    return ["yes" if r > threshold_value else "no" for r in residuals]

def calculate_forecastability_score(data,features):
    score = 0
    # Variance
    if 'std_deviation' in features and data.mean() != 0:
        cv = features['std_deviation'] / data.mean()
        score += 2.0 * min(1.0, 1.0/max(cv, 0.001))
    # trend strength
    if 'trend' in features:
        trend_norm = min(1.0, abs(features['trend']) / max(features['std_deviation'], 0.001))
        score += 3.0 * (1.0 - trend_norm)
    # 3. Seasonality Score
    if 'seasonality_strength' in features:
        score += 3.0 * min(1.0, max(0, features['seasonality_strength']))
    # 4. Randomness Score
    randomness = 0
    if 'skew' in features:
        randomness += 0.5 * min(1.0, abs(features['skew']))
    if 'kurtosis' in features:
        randomness += 0.5 * min(1.0, max(0, features['kurtosis'])/10)
    score += 2.0 * (1.0 - min(1.0, randomness))
    
    return min(10.0, max(0.0, round(score, 1)))

app = FastAPI()

@app.post('/predict')
async def predict_timeseries(
    file: UploadFile = File(...),
    date_from: str = Form(...),
    date_to: str = Form(...),
    test_data: str = Form(...)  # Receive as JSON string
):
    try:
        # Parse the test_data JSON string
        try:
            test_data_parsed = json.loads(test_data)
            test = pd.DataFrame(test_data_parsed)
            test['timestamp'] = pd.to_datetime(test['timestamp'])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid test data format: {str(e)}")
        # file extraction from request
        file_content = await file.read()
        
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext == 'csv':
            data = pd.read_csv(io.BytesIO(file_content))
        elif file_ext in ['xlsx', 'xls']:
            data = pd.read_excel(io.BytesIO(file_content))
        elif file_ext == 'txt':
            data = pd.read_csv(io.BytesIO(file_content), delimiter='\t')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")        
        # Data validation
        if 'point_timestamp' not in data.columns or 'point_value' not in data.columns:
            raise HTTPException(status_code=400, detail="File must contain 'point_timestamp' and 'point_value' columns")

        # Convert and validate test data
        try:
            test = pd.DataFrame(test_data)
            test['timestamp'] = pd.to_datetime(test['timestamp'])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid test data format: {str(e)}")
        test = test.rename(columns={'timestamp': 'timestamp_value', 'point_value': 'point_value'})
        #feature_extraction and predict label
        features = extract_features(data)
        features_array = np.array([list(features.values())])
        # total time calculation
        forcast_time = time.time()
        best_model_name = label_enc.inverse_transform(classifier.predict(features_array))
        
        # to filter data based on query parameter
        train_data = data[data['timestamp'] < pd.to_datetime(date_from)]
        if len(train_data) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient training data (need at least 10 historical points)"
            )
        # model prediction aand time calculation
        start_time = time.time()
        if(best_model_name == 'ets'): 
            mape, prediction = ets_model(train_data, test_data, features)
        elif(best_model_name == 'gru'): 
            mape, prediction = gru_model(train_data, test_data)
        elif(best_model_name == 'prophet'): 
            mape, prediction = prophet_model(train_data, test_data)
        elif(best_model_name == 'fourier'): 
            mape, prediction = fourier_model(train_data, test_data, features)
        else:
            mape, prediction = arima_model(train_data, test_data, features)
        time = time.time() - start_time
        anomalies = detect_anomalies(test['point_value'], prediction)
        forecastability_score = calculate_forecastability_score(train_data['point_value'],features)
        full_time = time.time() - forcast_time
        # Prepare result
        results = []
        for i in range(len(test)):
            results.append({
                "timestamp": test.iloc[i]['timestamp'].isoformat(),
                "point_value": float(test.iloc[i]['point_value']),
                "predicted": float(prediction[i]),
                "is_anomaly": anomalies[i]
            })
        
        # prepare response
        return JSONResponse({
            "forecastability_score": round(float(forecastability_score), 1),
            "number_of_batch_fits": len(train_data) // 100 + 1,
            "mape": float(mape),
            "avg_time_taken_per_fit_in_seconds": round(time, 2),
            "total_processing_time_seconds": round(full_time, 2),
            "best_model": best_model_name,
            "results": results
            })
    except Exception as e:
        raise HTTPException(500,str(e))