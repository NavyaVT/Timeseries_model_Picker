# Efficient Time Series Model Selection Algorithm

Time series analysis model performance is highly dependent on the intrinsic features of the dataset. This project implements an algorithm to select the most suitable model from a pool of five commonly used time series models:

1.  **ARIMA:** Autoregressive Integrated Moving Average
2.  **Prophet:** Developed by Facebook for forecasting with strong seasonality.
3.  **ETS:** Exponential Smoothing (Error, Trend, Seasonality)
4.  **GRU:** Gated Recurrent Unit (a type of Recurrent Neural Network)
5.  **MLPRegressor with Fourier Order:** Multi-layer Perceptron with Fourier terms for capturing seasonality.

Each of these models excels in capturing specific features like trends, seasonality, stationarity, and correlations.

## Approach

1.  **Model Evaluation & Feature Extraction:**
    * Multiple time series datasets are used.
    * All five models are applied to each dataset.
    * The Mean Absolute Percentage Error (MAPE) is calculated for each model's performance.
    * Statistical and seasonal features are extracted from the datasets.
    * A new dataset is created, combining the extracted features with the best performing model for each original time series.

2.  **Classifier Training:**
    * A **Random Forest Classifier** is used for model selection. This classifier is chosen for its:
        * Ability to perform well with limited data.
        * Resistance to overfitting.
        * Focus on predictive performance.
    * **GridSearchCV** is employed for hyperparameter tuning to optimize the Random Forest model.
    * **StratifiedShuffleSplit** is utilized for robust cross-validation, ensuring balanced representation of classes during training and validation.

## Usage Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/NavyaVT/Datagenie_DS](https://github.com/NavyaVT/Datagenie_DS)
    ```

2.  **Install the requirements file:**
    ```bash
     pip install -r requirements.txt
    ```
3.  **Run the FastAPI Backend:**
    * Navigate to the `core` folder.
    * Start the FastAPI server:
        ```bash
        uvicorn FastAPI:app --reload
        ```

4.  **Run the Streamlit Application:**
    * Run the Streamlit app:
        ```bash
        streamlit run streamlit_forecast.py
        ```

4.  **Prediction:**
    * Follow the instructions provided in the Streamlit application.
    * Input your time series data.
    * The application will predict the most suitable time series model and provide the test point features.
