import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def prediction_app():
    st.title("Prediksi Risiko Stroke")
    st.write("Masukkan data pasien untuk memprediksi risiko stroke menggunakan model Logistic Regression.")
    
    # 1. Load Model & Metadata
    model = joblib.load("model_stroke.pkl")
    feature_names = joblib.load("model_features.pkl")      # Fitur model (hasil get_dummies)
    numeric_cols = joblib.load("numeric_columns.pkl")      # Kolom numerik sebelum encoding

    # 2. Form Input Pengguna
    st.write("### Input Data Pasien")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.number_input("Age", 0, 120, 45)
    with col3:
        hypertension = st.selectbox("Hypertension", [0, 1])
    with col4:
        heart_disease = st.selectbox("Heart Disease", [0, 1])
    with col5:
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    with col2:
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    with col3:
        avg_glucose_level = st.number_input("Avg Glucose Level", 0.0, 300.0, 90.0)
    with col4:
        bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
    with col5:
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "smokes", "formerly smoked", "Unknown"])

    # 3. Kategori Otomatis
    col1, col2, col3 = st.columns(3)

    # Age Category
    with col1:
        if age <= 18:
            age_category = "Children"
            color = "#4DA6FF"
        elif age <= 35:
            age_category = "Young Adult"
            color = "#4CAF50"
        elif age <= 55:
            age_category = "Adult"
            color = "#EBD300"
        elif age <= 75:
            age_category = "Senior"
            color = "#F44336"
        else:
            age_category = "Elderly"
            color = "#8C007E"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Age Category</div>
                <div style="font-size:16px;font-weight:bold;">{age_category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    #BMI Category
    with col2:
        if bmi < 18.5:
            bmi_category = "Underweight"
            color = "#4DA6FF"
        elif bmi < 25:
            bmi_category = "Normal"
            color = "#4CAF50"
        elif bmi < 30:
            bmi_category = "Overweight"
            color = "#EBD300"
        else:
            bmi_category = "Obesity"
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">BMI Category</div>
                <div style="font-size:16px;font-weight:bold;">{bmi_category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    #Glucose Category
    with col3:
        if avg_glucose_level < 100:
            glucose_category = "Normal"
            color = "#4CAF50"
        elif avg_glucose_level < 126:
            glucose_category = "Prediabetes"
            color = "#EBD300"
        else:
            glucose_category = "Diabetes"
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Glucose Category</div>
                <div style="font-size:16px;font-weight:bold;">{glucose_category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    col4, col5, col6 = st.columns(3)
    #Chronic Count
    with col4:
        chronic_count = (hypertension + heart_disease + (avg_glucose_level >= 126))
        if chronic_count == 0:
            cat = 'Low Risk'
            color = "#4CAF50"
        elif chronic_count == 1:
            cat = 'Medium Risk'
            color = "#EBD300"
        else:
            cat = 'High Risk'
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Chronic Category</div>
                <div style="font-size:16px;font-weight:bold;">{cat}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    #Risk Score
    with col5:
        risk_score = (hypertension + heart_disease + (age > 55) + (avg_glucose_level >= 126) + (bmi >= 30))
        if risk_score <= 1:
            cat = 'Low Risk'
            color = "#4CAF50" 
        elif risk_score <= 3:
            cat = 'Medium Risk'
            color = "#EBD300" 
        else:
            cat = 'High Risk'
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Risk Score</div>
                <div style="font-size:16px;font-weight:bold;">{cat}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    #Lifestyle Risk
    with col6:
        if smoking_status == "smokes" and bmi_category == "Obesity":
            lifestyle_risk = "High Risk"
            color = "#F44336"
        elif smoking_status == "smokes" or bmi_category == "Overweight":
            lifestyle_risk = "Medium Risk"
            color = "#EBD300"
        else:
            lifestyle_risk = "Low Risk"
            color = "#4CAF50" 
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Lifestyle Score</div>
                <div style="font-size:16px;font-weight:bold;">{lifestyle_risk}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("") 

    # 3. Masukkan menjadi DataFrame
    user_df = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status],
        "age_category": [age_category],
        "bmi_category": [bmi_category],
        "glucose_category": [glucose_category],
        "chronic_count": [chronic_count],
        "risk_score": [risk_score],
        "lifestyle_risk": [lifestyle_risk]
    })

    # 4. One-Hot Encoding & Normalisasi
    user_processed = pd.get_dummies(user_df, drop_first=True)

    # Tambahkan kolom yang hilang (harus sama seperti training)
    for col in feature_names:
        if col not in user_processed.columns:
            user_processed[col] = 0

    # Pastikan urutan kolom sama persis
    user_processed = user_processed[feature_names]

    # Normalisasi numerik (MinMaxScaler baru)
    scaler = MinMaxScaler()
    for col in numeric_cols:
        user_processed[col] = scaler.fit_transform(user_processed[[col]])

    # 5. Prediksi
    if st.button("Prediksi Stroke"):
        prob = model.predict_proba(user_processed)[0][1]
        pred = model.predict(user_processed)[0]

        st.write("### 🔍 Hasil Prediksi")
        st.metric("Probabilitas Stroke", f"{prob*1000:.2f}%")

        probability_display = prob*1000

        if probability_display >= 50:
            st.error("⚠️ Pasien berisiko tinggi terkena **Stroke**.")
        else:
            st.success("✅ Pasien berada pada risiko **rendah** untuk terkena Stroke.")

        # Interpretasi tambahan
        st.write("---")
        st.write("### 📌 Interpretasi")
        st.write("""
        - **Probabilitas > 50%** → Model cenderung memprediksi *Stroke (1)*  
        - **Probabilitas < 50%** → Model cenderung memprediksi *Tidak Stroke (0)*  
        """)


# =========================
# 6. Panggil Fungsi Jika File Ini Dibuka
# =========================
if __name__ == "__main__":
    prediction_app()
