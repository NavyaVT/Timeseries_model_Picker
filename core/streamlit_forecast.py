import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Time Series Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stFileUploader>div>div>div>button {
            background-color: #2196F3;
            color: white;
        }
        .error-message {
            color: #ff4444;
            padding: 10px;
            border-radius: 5px;
            background-color: #ffebee;
        }
        .success-message {
            color: #00C851;
            padding: 10px;
            border-radius: 5px;
            background-color: #e8f5e9;
        }
        .header {
            color: #2E86AB;
        }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/predict"

def validate_test_data(test_data):
    """Validate the test data format"""
    if not test_data:
        return False, "Please add at least one test data point"
    
    for item in test_data:
        if not all(key in item for key in ['timestamp', 'point_value']):
            return False, "Each test point must contain 'timestamp' and 'point_value'"
        try:
            pd.to_datetime(item['timestamp'])
            float(item['point_value'])
        except (ValueError, TypeError):
            return False, "Invalid timestamp or point_value format"
    return True, ""

def display_results(response):
    """Display the prediction results in a formatted way"""
    st.subheader("📊 Forecast Results", divider="rainbow")
    
    # Metrics columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Forecastability Score", f"{response['forecastability_score']}/10")
    with col2:
        st.metric("Best Model", response['best_model'])
    with col3:
        st.metric("MAPE", f"{response['mape']:.2f}%")
    with col4:
        st.metric("Processing Time", f"{response['total_processing_time_seconds']:.2f}s")
    
    # Results table
    results_df = pd.DataFrame(response['results'])
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    results_df['error'] = results_df['point_value'] - results_df['predicted']
    
    st.dataframe(
        results_df.style.format({
            'point_value': '{:.2f}',
            'predicted': '{:.2f}',
            'error': '{:.2f}'
        }).applymap(
            lambda x: 'background-color: #ffdddd' if x == 'yes' else '', 
            subset=['is_anomaly']
        ),
        use_container_width=True
    )
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=results_df, x='timestamp', y='point_value', label='Actual', ax=ax)
    sns.lineplot(data=results_df, x='timestamp', y='predicted', label='Predicted', ax=ax)
    sns.scatterplot(
        data=results_df[results_df['is_anomaly'] == 'yes'],
        x='timestamp', y='point_value',
        color='red', label='Anomaly', s=100, ax=ax
    )
    y_min = min(results_df['point_value'].min(), results_df['predicted'].min())
    y_max = max(results_df['point_value'].max(), results_df['predicted'].max())
    margin = (y_max - y_min) * 0.05  # 5% margin
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_title('Actual vs Predicted Values')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    st.pyplot(fig)

def send_request(uploaded_file, test_data, date_from, date_to):
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
        payload = {
            'test_data': test_data,
            'date_from': date_from,
            'date_to': date_to,
        }
        headers = {} #content type is handeled by requests library.

        with st.spinner('🚀 Processing your forecast...'):
            response = requests.post(API_URL, files=files, data={'request_data':json.dumps(payload)})

        if response.status_code == 200:
            st.session_state.last_response = response.json()
            st.markdown(f"<div class='success-message'>✅ Forecast completed successfully!</div>", unsafe_allow_html=True)
            return response.json()
        else:
            error_msg = response.json().get('detail', 'Unknown error occurred')
            st.markdown(f"<div class='error-message'>❌ API Error: {error_msg}</div>", unsafe_allow_html=True)
            return None

    except requests.exceptions.RequestException as e:
        st.markdown(f"<div class='error-message'>❌ Connection Error: {str(e)}</div>", unsafe_allow_html=True)
        return None
    except Exception as e:
        st.markdown(f"<div class='error-message'>❌ Unexpected Error: {str(e)}</div>", unsafe_allow_html=True)
        return None

def main():
    st.title("⏳ Time Series Forecasting Dashboard")
    st.markdown("Upload your training data and test points to generate forecasts")
    
    with st.form("forecast_form"):
        # File upload section
        st.subheader("📁 Training Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV, Excel, or TXT file",
            type=['csv', 'xlsx', 'xls', 'txt'],
            help="File must contain 'point_timestamp' and 'point_value' columns"
        )
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("Training cutoff date", value=datetime.now())
        with col2:
            date_to = st.date_input("Forecast end date", value=datetime.now())
        
        # Test data input
        st.subheader("🧪 Test Data Input")
        test_data_json = st.text_area(
            "Enter test data as JSON array",
            value='[{"timestamp": "2023-01-01T00:00:00", "point_value": 100}]',
            height=150,
            help='Format: [{"timestamp": "...", "point_value": 123}, ...]'
        )
        
        submitted = st.form_submit_button("Generate Forecast")
    
    # Process form submission
    if submitted:
        if not uploaded_file:
            st.markdown("<div class='error-message'>❌ Please upload a training data file</div>", unsafe_allow_html=True)
            return
        
        try:
            test_data = json.loads(test_data_json)
            is_valid, validation_msg = validate_test_data(test_data)
            if not is_valid:
                st.markdown(f"<div class='error-message'>❌ {validation_msg}</div>", unsafe_allow_html=True)
                return
        except json.JSONDecodeError:
            st.markdown("<div class='error-message'>❌ Invalid JSON format for test data</div>", unsafe_allow_html=True)
            return
        
        response = send_request(uploaded_file, test_data, date_from.isoformat(), date_to.isoformat())
        if response:
            display_results(response)
    
    # Display last response if available
    if 'last_response' in st.session_state:
        with st.expander("📜 Last Forecast Results"):
            display_results(st.session_state.last_response)

if __name__ == "__main__":
    main()