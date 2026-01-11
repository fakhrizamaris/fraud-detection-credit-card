"""
Machine Learning Tab - Penjelasan proses training model
"""
import streamlit as st
import pandas as pd
import altair as alt


def render(model, feature_columns, load_data_func):
    """
    Render tab Machine Learning
    
    Args:
        model: Trained model
        feature_columns: List of feature column names
        load_data_func: Function to load dataset
    """
    st.title("Machine Learning Pipeline")
    st.markdown("### Proses Training Model Fraud Detection")
    st.markdown("---")
    
    # 1. Data Preprocessing
    st.markdown("## 1. Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Langkah Preprocessing
        
        **a. Penanganan Missing Values**
        - Mengidentifikasi dan mengisi nilai kosong
        - Strategi: Mode untuk kategorikal, Median untuk numerik
        
        **b. Feature Engineering**
        - Ekstraksi `hour` dari timestamp transaksi
        - Kalkulasi `age` dari tanggal lahir
        - Membuat `amt_per_hour_ratio` untuk deteksi anomali
        - Membuat `is_weekend` dari hari transaksi
        
        **c. Encoding Kategorikal**
        - Label Encoding untuk: `category`, `gender`, `state`
        - Menyimpan mapping untuk prediksi data baru
        """)
    
    with col2:
        st.markdown("""
        ### Normalisasi Data
        
        Fitur numerik dinormalisasi menggunakan **StandardScaler**:
        
        ```
        z = (x - μ) / σ
        ```
        
        Dimana:
        - `x` = nilai asli
        - `μ` = rata-rata (mean)
        - `σ` = standar deviasi
        
        **Fitur yang dinormalisasi:**
        - `amt` (jumlah transaksi)
        - `age` (usia)
        - `hour` (jam)
        - `amt_per_hour_ratio`
        """)
    
    st.markdown("---")
    
    # 2. Class Imbalance
    st.markdown("## 2. Penanganan Class Imbalance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Masalah Imbalance
        
        Dataset fraud detection biasanya **tidak seimbang**:
        - Transaksi **Normal**: ~99%
        - Transaksi **Fraud**: ~1%
        
        Tanpa penanganan, model akan:
        - Selalu memprediksi "Normal"
        - Gagal mendeteksi fraud
        - Memiliki Recall rendah
        """)
        
        # Class Distribution Visualization
        try:
            df_temp = load_data_func()
            class_counts = df_temp['is_fraud'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            class_counts['Class'] = class_counts['Class'].map({0: 'Normal', 1: 'Fraud'})
            
            pie_class = alt.Chart(class_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Class", type="nominal", scale=alt.Scale(
                    domain=['Normal', 'Fraud'],
                    range=['#4CAF50', '#F44336']
                )),
                tooltip=['Class', 'Count']
            ).properties(height=250, title='Distribusi Kelas (Sebelum SMOTE)')
            st.altair_chart(pie_class, width='stretch')
        except:
            st.info("Dataset tidak tersedia untuk visualisasi")
    
    with col2:
        st.markdown("""
        ### Solusi: SMOTE
        
        **Synthetic Minority Over-sampling Technique (SMOTE)**:
        
        1. Memilih sampel minoritas (Fraud)
        2. Mencari k-nearest neighbors
        3. Membuat sampel sintetis baru
        4. Menyeimbangkan distribusi kelas
        
        **Hasil:**
        - Kelas Normal: 50%
        - Kelas Fraud: 50%
        
        Model dapat belajar pola fraud dengan lebih baik!
        """)
        
        # After SMOTE Visualization
        balanced_data = pd.DataFrame({
            'Class': ['Normal', 'Fraud'],
            'Count': [5000, 5000]  # Simulated balanced
        })
        
        pie_balanced = alt.Chart(balanced_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Class", type="nominal", scale=alt.Scale(
                domain=['Normal', 'Fraud'],
                range=['#4CAF50', '#F44336']
            )),
            tooltip=['Class', 'Count']
        ).properties(height=250, title='Distribusi Kelas (Setelah SMOTE)')
        st.altair_chart(pie_balanced, width='stretch')
    
    st.markdown("---")
    
    # 3. Model Training
    st.markdown("## 3. Training Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Random Forest Classifier
        
        **Mengapa Random Forest?**
        - Robust terhadap overfitting
        - Dapat menangani data non-linear
        - Memberikan feature importance
        - Performa tinggi untuk klasifikasi
        
        **Hyperparameters:**
        """)
        
        st.code("""
model = RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=15,            # Kedalaman maksimal pohon
    min_samples_split=5,     # Minimum sampel untuk split
    min_samples_leaf=2,      # Minimum sampel di leaf
    random_state=42,         # Reproducibility
    n_jobs=-1,               # Gunakan semua CPU cores
    verbose=0                # Nonaktifkan verbose output
)
        """, language="python")
    
    with col2:
        st.markdown("""
        ### Proses Training
        
        **1. Train-Test Split**
        - Training: 80%
        - Testing: 20%
        - Stratified sampling
        
        **2. Cross-Validation**
        - 5-Fold CV
        - Mengevaluasi konsistensi performa
        
        **3. Metrik Evaluasi**
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - ROC-AUC
        """)
    
    st.markdown("---")
    
    # 4. Feature Importance Explanation
    st.markdown("## 4. Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        feature_imp_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Altair bar chart
        importance_chart = alt.Chart(feature_imp_df).mark_bar().encode(
            x=alt.X('Importance:Q', title='Skor Importance'),
            y=alt.Y('Feature:N', sort='-x', title='Fitur'),
            color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
        ).properties(height=300, title='Feature Importance - Random Forest')
        st.altair_chart(importance_chart, width='stretch')
        
        st.markdown("""
        **Interpretasi:**
        - **amt (Amount)**: Jumlah transaksi adalah prediktor terkuat
        - **hour**: Jam transaksi mempengaruhi probabilitas fraud
        - **age**: Usia pemegang kartu juga signifikan
        - **category**: Jenis merchant berpengaruh pada pola fraud
        """)
    
    st.markdown("---")
    st.success("Model telah ditraining dan siap digunakan untuk prediksi!")
