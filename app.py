"""
Fraud Detection System - Main Application
=========================================
Streamlit application untuk deteksi fraud kartu kredit menggunakan Machine Learning.

Struktur:
- tabs/about_dataset.py    : Tab informasi dataset
- tabs/fraud_detection.py  : Tab prediksi fraud
- tabs/dashboard.py        : Tab dashboard data
- tabs/machine_learning.py : Tab penjelasan ML pipeline
- tabs/model_performance.py: Tab evaluasi model
"""
import streamlit as st
import pandas as pd
import pickle

# Import tab modules
from tabs import about_dataset
from tabs import fraud_detection
from tabs import dashboard
from tabs import machine_learning
from tabs import model_performance
from tabs import contact_me

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Credit Card Fraud Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_model():
    """Load model dan preprocessors dari file pickle"""
    with open('models/fraud_detection_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_data():
    """Load dataset transaksi untuk visualisasi"""
    df = pd.read_csv('data/credit_card_transactions2.csv')
    return df

# Load model artifacts
try:
    model_artifacts = load_model()
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    label_encoders = model_artifacts['label_encoders']
    feature_columns = model_artifacts['feature_columns']
    numerical_cols = model_artifacts['numerical_cols']
    
    # Extract model info if available
    model_info = model_artifacts.get('model_info', {})
    performance = model_artifacts.get('performance', {})
except FileNotFoundError:
    st.error("❌ Model belum di-training! Jalankan `training_model.py` terlebih dahulu.")
    st.stop()

# ========================================
# INITIALIZE SESSION STATE
# ========================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ========================================
# MAIN HEADER
# ========================================
st.markdown("""
<div style="text-align: center; padding: 20px; margin-bottom: 10px;">
    <h1 style="margin: 0;">Credit Card Fraud Analysis</h1>
    <p style="margin: 5px 0 0 0;">Machine Learning-Based Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# TABS
# ========================================
tab_about, tab1, tab2, tab3, tab4, tab_contact = st.tabs([
    "About Dataset", 
    "Dashboard", 
    "Fraud Detection", 
    "Machine Learning", 
    "Model Performance",
    "Contact Me"
])

# ========================================
# RENDER TABS
# ========================================
with tab_about:
    about_dataset.render()

with tab1:
    dashboard.render(load_data_func=load_data)

with tab2:
    fraud_detection.render(
        model=model,
        scaler=scaler,
        label_encoders=label_encoders,
        feature_columns=feature_columns,
        numerical_cols=numerical_cols
    )

with tab3:
    machine_learning.render(
        model=model,
        feature_columns=feature_columns,
        load_data_func=load_data
    )

with tab4:
    model_performance.render(
        model=model,
        model_info=model_info,
        performance=performance,
        feature_columns=feature_columns
    )

with tab_contact:
    contact_me.render()

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🛡️ Credit Card Fraud Analysis System | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
