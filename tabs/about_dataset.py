"""
About Dataset Tab - Informasi tentang dataset dan sistem deteksi fraud
"""
import streamlit as st


def render():
    """Render tab About Dataset"""
    st.title("About Dataset & Fraud Detection")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Latar Belakang
        
        **Fraud kartu kredit** merupakan salah satu kejahatan finansial yang paling 
        merugikan di era digital. Tantangan utama yang dihadapi meliputi:
        
        - Kerugian finansial yang signifikan bagi konsumen dan institusi
        - Teknik fraud yang semakin canggih
        - Metode deteksi manual yang lambat dan tidak efisien
        
        Sistem deteksi fraud berbasis **Machine Learning** dapat mengidentifikasi 
        transaksi mencurigakan dalam **hitungan milidetik**, secara signifikan 
        mengurangi kerugian finansial dan meningkatkan keamanan.
        """)
        
        st.markdown("""
        ### Tentang Dataset
        
        Dataset ini berisi **transaksi kartu kredit** dengan fitur-fitur berikut:
        
        | Fitur | Deskripsi |
        |-------|-----------|
        | `category` | Jenis merchant (grocery, gas, shopping, dll.) |
        | `amt` | Jumlah transaksi dalam USD |
        | `gender` | Jenis kelamin pemegang kartu |
        | `state` | Negara bagian AS tempat transaksi terjadi |
        | `age` | Usia pemegang kartu |
        | `hour` | Jam transaksi dilakukan |
        | `is_weekend` | Apakah transaksi di akhir pekan |
        | `is_fraud` | Label (0 = Normal, 1 = Fraud) |
        """)
    
    with col2:
        st.markdown("""
        ### Metodologi
        
        Sistem ini menggunakan algoritma **Random Forest Classifier** dengan proses:
        
        1. **Preprocessing Data**
           - Encoding variabel kategorikal
           - Normalisasi fitur numerik
           - Feature engineering (amt_per_hour_ratio)
        
        2. **Training Model**
           - Pembagian data: 80% training, 20% testing
           - Hyperparameter tuning
           - Cross-validation
        
        3. **Evaluasi**
           - Accuracy, Precision, Recall, F1-Score
           - ROC-AUC untuk mengukur diskriminasi model
        """)
        
        st.markdown("""
        ### Manfaat Sistem
        
        - **Deteksi Real-time**: Analisis transaksi dalam hitungan detik
        - **Akurasi Tinggi**: Model terlatih dengan ribuan data historis
        - **Pencegahan Kerugian**: Identifikasi fraud sebelum terjadi
        - **Efisiensi Operasional**: Mengurangi beban review manual
        """)
    
    st.markdown("---")
    st.info("**Tips**: Gunakan tab **Fraud Detection** untuk menganalisis transaksi baru, atau lihat **Dashboard** untuk eksplorasi data historis.")
