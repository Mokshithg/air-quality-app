import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --- Settings ---
st.set_page_config(
    page_title="AirSage Pro",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_FEATURES = [
    'PT08.S1(CO)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)',
    'PT08.S3(NOx)', 'T', 'RH', 'AH', 'Hour', 'Month', 'DayOfWeek'
]

# --- Model Load ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/air_quality_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# Get feature names (fallback to defaults if model doesn't have them)
features = getattr(model, 'feature_names_in_', DEFAULT_FEATURES) if model else DEFAULT_FEATURES

# --- Gauge Chart Function ---
def create_gauge(value, min_val=0, max_val=15, threshold=9.4):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, 4.4], 'color': "lightgreen"},
                {'range': [4.4, threshold], 'color': "orange"},
                {'range': [threshold, max_val], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(margin=dict(t=0, b=0))
    return fig

# --- Sidebar ---
with st.sidebar:
    st.title("Configuration")
    st.markdown("---")
    alert_threshold = st.slider("Alert Threshold (mg/m³)", 4.4, 15.0, 9.4)
    st.markdown(f"""
    ### Expected Features:
    {", ".join(features)}
    """)

# --- Main Interface ---
st.title("🌫️ AirSage Pro")
st.caption("Industrial-Grade Air Quality Monitoring System")

tab1, tab2 = st.tabs(["Prediction", "Documentation"])

with tab1:
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Pollutant Sensors")
            pt08_s1_co = st.number_input('PT08.S1(CO)', value=1000.0, step=0.1)
            nmhc_gt = st.number_input('NMHC(GT)', value=200.0, step=0.1)
            nox_gt = st.number_input('NOx(GT)', value=150.0, step=0.1)
            no2_gt = st.number_input('NO2(GT)', value=50.0, step=0.1)
            pt08_s3_nox = st.number_input('PT08.S3(NOx)', value=800.0, step=0.1)
            
        with col2:
            st.subheader("Environmental Data")
            temperature = st.slider('Temperature (T)', -20.0, 50.0, 20.0)
            humidity = st.slider('Humidity (RH)', 0.0, 100.0, 50.0)
            abs_humidity = st.number_input('Abs Humidity (AH)', value=1.0, step=0.01)
            
        with col3:
            st.subheader("Temporal Data")
            current_time = datetime.now()
            hour = st.slider('Hour', 0, 23, current_time.hour)
            month = st.selectbox('Month', range(1,13), current_time.month-1)
            day_of_week = st.selectbox('Day of Week', range(7), 0)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded - cannot make predictions")
        else:
            input_data = {
                'PT08.S1(CO)': [pt08_s1_co],
                'NMHC(GT)': [nmhc_gt],
                'NOx(GT)': [nox_gt],
                'NO2(GT)': [no2_gt],
                'PT08.S3(NOx)': [pt08_s3_nox],
                'T': [temperature],
                'RH': [humidity],
                'AH': [abs_humidity],
                'Hour': [hour],
                'Month': [month],
                'DayOfWeek': [day_of_week]
            }
            
            # Ensure we only use features the model expects
            input_df = pd.DataFrame(input_data)[features]
            
            try:
                prediction = model.predict(input_df)[0]
                
                with st.container():
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1,3,1])
                    
                    with col2:
                        st.plotly_chart(
                            create_gauge(prediction, threshold=alert_threshold), 
                            use_container_width=True
                        )
                        
                        if prediction > alert_threshold:
                            st.error("🚨 Critical Alert: Evacuation recommended!")
                            st.snow()
                        elif prediction > 4.4:
                            st.warning("⚠️ Caution: Sensitive groups should reduce exposure")
                        else:
                            st.success("✅ Air Quality Normal")
                            st.balloons()
                        
                        st.metric("Predicted CO Concentration", f"{prediction:.2f} mg/m³", 
                                delta_color="inverse")
                            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                with st.expander("Debug Details"):
                    st.write("Input Features:", input_df.columns.tolist())
                    st.write("Input Values:", input_df.values.tolist())
                    st.write("Model Features:", features)

with tab2:
    st.subheader("Technical Documentation")
    st.markdown("""
    ### Model Specifications
    - **Type**: Linear Regression
    - **Input Features**: 
        {}
    
    ### Safety Thresholds
    | Level       | CO (mg/m³) | Health Impact |
    |-------------|------------|---------------|
    | Safe        | < 4.4      | Normal conditions |
    | Moderate    | 4.4-9.4    | Sensitive groups affected |
    | Hazardous   | > 9.4      | Health warnings issued |
    """.format(", ".join(features)))