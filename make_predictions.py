import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.deterministic import Fourier, DeterministicProcess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
logger = logging.getLogger("cmdstanpy")
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.NullHandler())

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure data has correct columns and formatting"""
    if 'point_timestamp' not in data.columns or 'point_value' not in data.columns:
        raise ValueError("Data must contain 'point_timestamp' and 'point_value' columns")
    
    data = data.copy()
    data['point_timestamp'] = pd.to_datetime(data['point_timestamp'])
    data = data.set_index('point_timestamp')
    return data.sort_index()

def ets_model(train: pd.DataFrame, test: pd.DataFrame, features: dict):
    """Exponential Smoothing State Space Model"""
    try:
        train_vals = train['point_value'].values
        test_vals = test['point_value'].values
        
        # Determine ETS model parameters
        err = 'add'
        trend = 'add' if features['trend'] > 0 else None
        season = 'add' if features['seasonality_strength'] > 0.1 else None
        period = features['period'] if len(train) >= (2*features['period']) else None
        
        model = ETSModel(train_vals, 
                        error=err, 
                        trend=trend, 
                        seasonal=season, 
                        seasonal_periods=period).fit()
        
        predictions = model.forecast(steps=len(test_vals))
        return calculate_mape(test_vals, predictions), predictions
    
    except Exception as e:
        print(f"ETS Error: {e}")
        return float('inf'), np.zeros_like(test['point_value'])

def prophet_model(train: pd.DataFrame, test: pd.DataFrame):
    """Facebook Prophet Model"""
    try:
        # Prepare dataframes with expected column names
        train_df = train.reset_index()[['point_timestamp', 'point_value']]
        train_df.columns = ['ds', 'y']
        
        model = Prophet()
        model.fit(train_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(
            periods=len(test), 
            include_history=False
        )
        
        forecast = model.predict(future)
        predictions = forecast['yhat'].values
        test_vals = test['point_value'].values
        
        return calculate_mape(test_vals, predictions), predictions
    
    except Exception as e:
        print(f"Prophet Error: {e}")
        return float('inf'), np.zeros_like(test['point_value'])

def fourier_model(train: pd.DataFrame, test: pd.DataFrame, features: dict):
    """Fourier Features with MLP Model"""
    try:
        y_train = train['point_value'].values
        y_test = test['point_value'].values
        
        # Determine optimal Fourier order
        fourier_order = 2  # default
        if features['period'] <= 7:
            fourier_order = 3 if features['seasonality_strength'] > 0.7 else 2
        elif features['period'] <= 30:
            fourier_order = 4 if (len(train) > 200 and features['seasonality_strength'] > 0.6) else 2
        elif features['period'] > 30:
            fourier_order = 5 if features['acf1'] > 0.7 else 3
        
        # Create Fourier features
        fourier = Fourier(period=features['period'], order=fourier_order)
        dp = DeterministicProcess(
            index=train.index,
            constant=False,
            order=0,
            seasonal=True,
            additional_terms=[fourier]
        )
        
        # Train model
        X_train = dp.in_sample()
        model = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation='relu',
            solver='adam',
            max_iter=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict
        X_test = dp.out_of_sample(steps=len(test))
        predictions = model.predict(X_test)
        
        return calculate_mape(y_test, predictions), predictions
    
    except Exception as e:
        print(f"Fourier Error: {e}")
        return float('inf'), np.zeros_like(test['point_value'])

class GRUModel(nn.Module):
    """GRU Neural Network Model"""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 8, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(8, 1)
        
    def forward(self, x):
        x, _ = self.gru(x)
        return self.fc(self.dropout(x[:, -1, :])).squeeze()

def gru_model(train: pd.DataFrame, test: pd.DataFrame):
    """GRU Neural Network Forecasting"""
    try:
        y_train = train['point_value'].values
        y_test = test['point_value'].values
        
        # Validate data
        if len(y_train) < 15:
            return float('inf'), np.zeros_like(y_test)
            
        y_min = min(y_train.min(), y_test.min())
        y_max = max(y_train.max(), y_test.max())
        
        if y_max == y_min:
            return float('inf'), np.zeros_like(y_test)
        
        # Normalize data
        y_train_norm = (y_train - y_min) / (y_max - y_min)
        y_test_norm = (y_test - y_min) / (y_max - y_min)
        
        # Create sequences
        seq_len = min(3, len(y_train) // 5)
        if len(y_train) <= seq_len:
            return float('inf'), np.zeros_like(y_test)
            
        X_train, y_train_seq = [], []
        for i in range(len(y_train_norm) - seq_len):
            X_train.append(y_train_norm[i:i+seq_len])
            y_train_seq.append(y_train_norm[i+seq_len])
            
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32)
        
        # Initialize model
        model = GRUModel()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        # Training loop
        best_mape = float('inf')
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = F.mse_loss(outputs, y_train_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                train_preds = model(X_train).clamp(0, 1)
                current_mape = calculate_mape(
                    y_train_seq.numpy(), 
                    train_preds.numpy()
                )
                scheduler.step(current_mape)
                
                if current_mape < best_mape:
                    best_mape = current_mape
                    if best_mape < 1.0:  # Early stopping
                        break
        
        # Generate predictions
        model.eval()
        predictions_norm = []
        current_input = y_train_norm[-seq_len:].reshape(1, -1, 1)
        current_input = torch.tensor(current_input, dtype=torch.float32)
        
        for _ in range(len(y_test)):
            pred = model(current_input).item()
            predictions_norm.append(pred)
            current_input = torch.cat([
                current_input[:, 1:, :], 
                torch.tensor([[[pred]]], dtype=torch.float32)
            ], dim=1)
        
        # Denormalize predictions
        predictions = np.array(predictions_norm) * (y_max - y_min) + y_min
        return calculate_mape(y_test, predictions), predictions
        
    except Exception as e:
        print(f"GRU Error: {e}")
        return float('inf'), np.zeros_like(test['point_value'])