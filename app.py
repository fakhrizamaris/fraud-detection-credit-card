import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Fraud Detection System",
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
# TABS
# ========================================
tab1, tab2 = st.tabs(["🔍 Fraud Detection", "📊 Model Performance"])

# ========================================
# TAB 1: FRAUD DETECTION
# ========================================
with tab1:
    st.title("🛡️ Fraud Detection System")
    st.markdown("### Early Warning System untuk Deteksi Transaksi Mencurigakan")
    st.markdown("---")
    
    # ========================================
    # SIDEBAR - INPUT FORM
    # ========================================
    st.sidebar.header("📝 Input Data Transaksi")
    st.sidebar.markdown("Masukkan detail transaksi untuk analisis:")
    
    # Input Category
    category_options = list(label_encoders['category'].classes_)
    category = st.sidebar.selectbox(
        "Kategori Transaksi",
        options=category_options,
        help="Jenis merchant/toko tempat transaksi"
    )
    
    # Input Amount
    amt = st.sidebar.number_input(
        "Jumlah Transaksi (USD)",
        min_value=0.01,
        max_value=100000.0,
        value=50.0,
        step=10.0,
        help="Total nilai transaksi dalam Dollar"
    )
    
    # Input Gender
    gender_options = list(label_encoders['gender'].classes_)
    gender = st.sidebar.selectbox(
        "Jenis Kelamin",
        options=gender_options
    )
    
    # Input State
    state_options = list(label_encoders['state'].classes_)
    state = st.sidebar.selectbox(
        "Negara Bagian",
        options=state_options,
        help="Lokasi transaksi dilakukan"
    )
    
    # Input Age
    age = st.sidebar.slider(
        "Umur Pemegang Kartu",
        min_value=18,
        max_value=100,
        value=35,
        help="Umur user saat transaksi"
    )
    
    # Input Hour
    hour = st.sidebar.slider(
        "Jam Transaksi",
        min_value=0,
        max_value=23,
        value=14,
        help="Jam transaksi dilakukan (format 24 jam)"
    )
    
    # Input Weekend
    is_weekend = st.sidebar.checkbox(
        "Transaksi di Akhir Pekan?",
        value=False,
        help="Centang jika transaksi dilakukan Sabtu/Minggu"
    )
    
    st.sidebar.markdown("---")
    
    # ========================================
    # PREDIKSI
    # ========================================
    if st.sidebar.button("🔍 ANALISIS TRANSAKSI", type="primary", use_container_width=True):
        
        # Prepare input data
        amt_per_hour_ratio = amt / (hour + 1)
        
        # Encode categorical variables
        category_encoded = label_encoders['category'].transform([category])[0]
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        state_encoded = label_encoders['state'].transform([state])[0]
        
        # Buat dataframe
        input_data = pd.DataFrame({
            'category': [category_encoded],
            'amt': [amt],
            'gender': [gender_encoded],
            'state': [state_encoded],
            'age': [age],
            'hour': [hour],
            'is_weekend': [int(is_weekend)],
            'amt_per_hour_ratio': [amt_per_hour_ratio]
        })
        
        # Reorder
        input_data = input_data[feature_columns]
        
        # Scaling
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Prediksi
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        confidence = prediction_proba[prediction] * 100
        
        # Save to history
        prediction_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'amount': amt,
            'category': category,
            'state': state,
            'hour': hour,
            'prediction': 'FRAUD' if prediction == 1 else 'SAFE',
            'confidence': confidence,
            'prob_safe': prediction_proba[0] * 100,
            'prob_fraud': prediction_proba[1] * 100
        }
        st.session_state.prediction_history.append(prediction_record)
        
        # ========================================
        # TAMPILKAN HASIL
        # ========================================
        st.markdown("## 📊 Hasil Analisis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Transaksi", f"${amt:,.2f}")
        with col2:
            st.metric("Jam Transaksi", f"{hour}:00")
        with col3:
            st.metric("Lokasi", state)
        
        st.markdown("---")
        
        # Output prediksi
        if prediction == 0:
            st.success("### ✅ TRANSAKSI AMAN")
            st.markdown(f"**Confidence Level:** {confidence:.2f}%")
            st.info("💡 Transaksi ini tidak menunjukkan pola mencurigakan. Dapat diproses dengan normal.")
            
        else:
            st.error("### 🚨 POTENSI INDIKASI FRAUD TERDETEKSI!")
            st.markdown(f"**Tingkat Keyakinan:** {confidence:.2f}%")
            st.warning("⚠️ **TINDAKAN YANG DISARANKAN:**")
            st.markdown("""
            - Lakukan verifikasi tambahan dengan pemegang kartu
            - Periksa histori transaksi sebelumnya
            - Aktifkan notifikasi ke tim fraud prevention
            - Jangan proses transaksi tanpa konfirmasi
            """)
        
        # Pie Chart Visualization
        st.markdown("---")
        st.markdown("### 📈 Distribusi Probabilitas")
        
        viz_col1, viz_col2 = st.columns([1, 1])
        
        with viz_col1:
            # Pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = ['#2ecc71', '#e74c3c']  # Green for Safe, Red for Fraud
            explode = (0.05, 0.05)
            
            ax.pie([prediction_proba[0], prediction_proba[1]], 
                   labels=['Safe', 'Fraud'],
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=colors,
                   explode=explode,
                   textprops={'fontsize': 12, 'weight': 'bold'})
            ax.set_title('Probability Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with viz_col2:
            st.markdown("#### 📋 Detail Probabilitas")
            st.metric("Probabilitas AMAN", f"{prediction_proba[0]*100:.2f}%", 
                     delta=f"{prediction_proba[0]*100 - 50:.1f}%" if prediction_proba[0] > 0.5 else None)
            st.metric("Probabilitas FRAUD", f"{prediction_proba[1]*100:.2f}%",
                     delta=f"{prediction_proba[1]*100 - 50:.1f}%" if prediction_proba[1] > 0.5 else None,
                     delta_color="inverse")
            
            # Progress bar
            st.markdown("**Risk Level:**")
            st.progress(prediction_proba[1])
        
        # Faktor risiko
        st.markdown("---")
        st.markdown("### 🎯 Faktor Analisis")
        
        risk_factors = []
        if amt > 500:
            risk_factors.append("💰 Transaksi bernilai tinggi (>$500)")
        if hour < 6 or hour > 22:
            risk_factors.append("🌙 Transaksi di jam tidak biasa (tengah malam/dini hari)")
        if is_weekend:
            risk_factors.append("📅 Transaksi di akhir pekan")
        if category in ['gas_transport', 'misc_net', 'shopping_net']:
            risk_factors.append("🏪 Kategori dengan risiko fraud lebih tinggi")
        
        if risk_factors:
            st.warning("**Faktor yang mempengaruhi analisis:**")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.info("✅ Tidak ada faktor risiko signifikan terdeteksi")
        
        # Download section
        st.markdown("---")
        st.markdown("### 📥 Download Hasil")
        
        # Prepare download data
        download_data = pd.DataFrame([prediction_record])
        csv = download_data.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="💾 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name=f'fraud_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )

# ========================================
# TAB 2: MODEL PERFORMANCE
# ========================================
with tab2:
    st.title("📊 Model Performance Dashboard")
    st.markdown("### Evaluasi Performa Model Random Forest")
    st.markdown("---")
    
    # Model Info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤖 Model Information")
        st.markdown(f"**Algorithm:** {model_info.get('algorithm', 'Random Forest')}")
        st.markdown(f"**N Estimators:** {model_info.get('n_estimators', 200)}")
        st.markdown(f"**Max Depth:** {model_info.get('max_depth', 15)}")
        if 'trained_at' in model_info:
            st.markdown(f"**Trained:** {model_info['trained_at']}")
    
    with col2:
        st.markdown("#### 📈 Performance Metrics")
        if performance:
            st.metric("Accuracy", f"{performance.get('accuracy', 0)*100:.2f}%")
            st.metric("Recall", f"{performance.get('recall', 0)*100:.2f}%")
            st.metric("Precision", f"{performance.get('precision', 0)*100:.2f}%")
        else:
            st.info("Performance metrics not available in model file")
    
    with col3:
        st.markdown("#### 🎯 Model Status")
        if performance:
            acc = performance.get('accuracy', 0)
            rec = performance.get('recall', 0)
            
            if acc >= 0.85 and rec >= 0.80:
                st.success("✅ Model Meets Requirements")
            else:
                st.warning("⚠️ Model Below Target")
            
            st.metric("F1-Score", f"{performance.get('f1_score', 0)*100:.2f}%")
            st.metric("ROC-AUC", f"{performance.get('roc_auc', 0):.4f}")
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🔍 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        feature_imp_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show table
        st.markdown("#### 📋 Feature Importance Table")
        st.dataframe(feature_imp_df.style.format({'Importance': '{:.4f}'}), use_container_width=True)
    
    st.markdown("---")
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("### 📜 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Download history
        csv_history = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download All Predictions (CSV)",
            data=csv_history,
            file_name=f'fraud_prediction_history_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    else:
        st.info("📭 Belum ada riwayat prediksi. Lakukan prediksi di tab 'Fraud Detection' terlebih dahulu.")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🔒 Fraud Detection System v1.0 | Powered by Random Forest & Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
