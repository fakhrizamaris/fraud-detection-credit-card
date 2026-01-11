"""
Fraud Detection Tab - Input form and fraud prediction
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def render(model, scaler, label_encoders, feature_columns, numerical_cols):
    """
    Render tab Fraud Detection
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        label_encoders: Dict of label encoders
        feature_columns: List of feature column names
        numerical_cols: List of numerical column names
    """
    st.title("Fraud Detection System")
    st.markdown("### Sistem Peringatan Dini untuk Deteksi Transaksi Mencurigakan")
    st.markdown("---")
    
    # ========================================
    # SIDEBAR - INPUT FORM
    # ========================================
    st.sidebar.header("Input Transaksi")
    st.sidebar.markdown("Masukkan detail transaksi untuk dianalisis:")
    
    # Input Category
    category_options = list(label_encoders['category'].classes_)
    
    def format_category(cat_name):
        return cat_name.replace('_', ' ').title()
    
    category = st.sidebar.selectbox(
        "Kategori Transaksi",
        options=category_options,
        format_func=format_category,
        help="Jenis merchant/toko"
    )
    
    # Input Amount
    amt = st.sidebar.number_input(
        "Jumlah Transaksi (USD)",
        min_value=0.01,
        max_value=100000.0,
        value=50.0,
        step=10.0,
        help="Total nilai transaksi dalam USD"
    )
    
    # Input Gender
    gender_options = list(label_encoders['gender'].classes_)
    
    gender_map = {
        'M': 'Laki-laki',
        'F': 'Perempuan'
    }
    
    gender = st.sidebar.selectbox(
        "Jenis Kelamin",
        options=gender_options,
        format_func=lambda x: gender_map.get(x, x),
        help="Jenis kelamin pemegang kartu"
    )
    
    # Input State
    state_options = list(label_encoders['state'].classes_)
    
    us_state_map = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    state = st.sidebar.selectbox(
        "Negara Bagian",
        options=state_options,
        format_func=lambda x: f"{x} - {us_state_map.get(x, x)}" if x in us_state_map else x,
        help="Lokasi transaksi"
    )
    
    # Input Age
    age = st.sidebar.slider(
        "Usia Pemegang Kartu",
        min_value=18,
        max_value=100,
        value=35,
        help="Usia pemegang kartu"
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
    
    # PREDIKSI
    analyze_clicked = st.sidebar.button("ANALISIS TRANSAKSI", type="primary", width='stretch')
    
    if analyze_clicked:
        
        # Prepare input data
        amt_per_hour_ratio = amt / (hour + 1)
        
        # Encode categorical variables
        category_encoded = label_encoders['category'].transform([category])[0]
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        state_encoded = label_encoders['state'].transform([state])[0]
        
        # Create dataframe
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
        
        # Prediction
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
        # DISPLAY RESULTS
        # ========================================
        st.markdown("## Hasil Analisis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Transaksi", f"${amt:,.2f}")
        with col2:
            st.metric("Jam Transaksi", f"{hour}:00")
        with col3:
            st.metric("Lokasi", state)
        
        # ========================================
        # RISK CATEGORY BADGES
        # ========================================
        st.markdown("### Kategori Risiko")
        badge_col1, badge_col2, badge_col3, badge_col4 = st.columns(4)
        
        # Professional color palette
        color_low = "#2d5a3d"      # Muted green
        color_medium = "#7a6c3e"   # Muted gold/brown
        color_high = "#8b5a3c"     # Muted orange/brown
        color_critical = "#6b3a3a" # Muted red/maroon
        color_neutral = "#4a5568"  # Slate gray
        
        # Amount Category
        with badge_col1:
            if amt < 50:
                amt_cat = "Low"
                bg_color = color_low
            elif amt < 200:
                amt_cat = "Medium"
                bg_color = color_medium
            elif amt < 500:
                amt_cat = "High"
                bg_color = color_high
            else:
                amt_cat = "Very High"
                bg_color = color_critical
            st.markdown(
                f"""
                <div style="background:{bg_color}; padding:15px; border-radius:8px; text-align:center; 
                            border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size:11px; color:#ccc; text-transform:uppercase; letter-spacing:1px;">Amount Risk</div>
                    <div style="font-size:16px; font-weight:600; color:#fff; margin-top:5px;">{amt_cat}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Time Category
        with badge_col2:
            if 6 <= hour <= 22:
                time_cat = "Normal"
                bg_color = color_low
            elif 22 < hour or hour < 2:
                time_cat = "Late Night"
                bg_color = color_high
            else:
                time_cat = "Early Morning"
                bg_color = color_medium
            st.markdown(
                f"""
                <div style="background:{bg_color}; padding:15px; border-radius:8px; text-align:center;
                            border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size:11px; color:#ccc; text-transform:uppercase; letter-spacing:1px;">Time Risk</div>
                    <div style="font-size:16px; font-weight:600; color:#fff; margin-top:5px;">{time_cat}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Age Category
        with badge_col3:
            if age < 25:
                age_cat = "Young Adult"
                bg_color = color_neutral
            elif age < 40:
                age_cat = "Adult"
                bg_color = color_neutral
            elif age < 60:
                age_cat = "Middle Age"
                bg_color = color_neutral
            else:
                age_cat = "Senior"
                bg_color = color_neutral
            st.markdown(
                f"""
                <div style="background:{bg_color}; padding:15px; border-radius:8px; text-align:center;
                            border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size:11px; color:#ccc; text-transform:uppercase; letter-spacing:1px;">Age Group</div>
                    <div style="font-size:16px; font-weight:600; color:#fff; margin-top:5px;">{age_cat}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Weekend Category
        with badge_col4:
            if is_weekend:
                weekend_cat = "Weekend"
                bg_color = color_medium
            else:
                weekend_cat = "Weekday"
                bg_color = color_low
            st.markdown(
                f"""
                <div style="background:{bg_color}; padding:15px; border-radius:8px; text-align:center;
                            border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size:11px; color:#ccc; text-transform:uppercase; letter-spacing:1px;">Day Type</div>
                    <div style="font-size:16px; font-weight:600; color:#fff; margin-top:5px;">{weekend_cat}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("")
        st.markdown("---")
        
        # Prediction output
        if prediction == 0:
            st.success("### ✅ TRANSAKSI AMAN")
            st.markdown(f"**Tingkat Keyakinan:** {confidence:.2f}%")
            st.info("Transaksi ini tidak menunjukkan pola mencurigakan. Dapat diproses dengan normal.")
            
        else:
            st.error("### POTENSI FRAUD TERDETEKSI!")
            st.markdown(f"**Tingkat Keyakinan:** {confidence:.2f}%")
            st.warning("**TINDAKAN YANG DISARANKAN:**")
            st.markdown("""
            - Lakukan verifikasi tambahan dengan pemegang kartu
            - Periksa riwayat transaksi sebelumnya
            - Notifikasi tim pencegahan fraud
            - Jangan proses transaksi tanpa konfirmasi
            """)
        
        # Pie Chart Visualization
        st.markdown("---")
        st.markdown("### Distribusi Probabilitas")
        
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
            st.markdown("#### Detail Probabilitas")
            st.metric("Probabilitas Aman", f"{prediction_proba[0]*100:.2f}%", 
                     delta=f"{prediction_proba[0]*100 - 50:.1f}%" if prediction_proba[0] > 0.5 else None)
            st.metric("Probabilitas Fraud", f"{prediction_proba[1]*100:.2f}%",
                     delta=f"{prediction_proba[1]*100 - 50:.1f}%" if prediction_proba[1] > 0.5 else None,
                     delta_color="inverse")
            
            # Progress bar
            st.markdown("**Level Risiko:**")
            st.progress(prediction_proba[1])
        
        # Risk factors
        st.markdown("---")
        st.markdown("### Faktor Analisis")
        
        risk_factors = []
        if amt > 500:
            risk_factors.append("Transaksi bernilai tinggi (>$500)")
        if hour < 6 or hour > 22:
            risk_factors.append("Jam transaksi tidak biasa (tengah malam/dini hari)")
        if is_weekend:
            risk_factors.append("Transaksi di akhir pekan")
        if category in ['gas_transport', 'misc_net', 'shopping_net']:
            risk_factors.append("Kategori dengan risiko fraud lebih tinggi")
        
        if risk_factors:
            st.warning("**Faktor yang mempengaruhi analisis:**")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.info("Tidak ada faktor risiko signifikan terdeteksi")
        
        # Download section
        st.markdown("---")
        st.markdown("### Unduh Hasil")
        
        # Prepare download data
        download_data = pd.DataFrame([prediction_record])
        csv = download_data.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Unduh Hasil Prediksi (CSV)",
            data=csv,
            file_name=f'fraud_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            width='stretch'
        )
    
    else:
        # ========================================
        # INSTRUCTIONS (shown before analysis)
        # ========================================
        st.info("**Silakan isi detail transaksi di sidebar sebelah kiri dan klik 'ANALISIS TRANSAKSI' untuk memulai analisis.**")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Cara Penggunaan
            
            **Langkah 1:** Masukkan detail transaksi di **sidebar** sebelah kiri:
            - Pilih kategori transaksi
            - Masukkan jumlah transaksi (USD)
            - Pilih jenis kelamin pemegang kartu
            - Pilih negara bagian AS
            - Atur usia pemegang kartu
            - Atur jam transaksi
            - Centang jika transaksi di akhir pekan
            
            **Langkah 2:** Klik tombol **"ANALISIS TRANSAKSI"**
            
            **Langkah 3:** Lihat hasil analisis meliputi:
            - Badge kategori risiko
            - Probabilitas fraud
            - Faktor risiko yang berkontribusi
            """)
        
        with col2:
            st.markdown("""
            ### Fitur yang Dianalisis
            
            | Fitur | Deskripsi |
            |-------|----------|
            | **Category** | Jenis merchant (grocery, gas, dll.) |
            | **Amount** | Nilai transaksi dalam USD |
            | **Gender** | Jenis kelamin pemegang kartu |
            | **State** | Negara bagian tempat transaksi |
            | **Age** | Usia pemegang kartu |
            | **Hour** | Jam transaksi (0-23) |
            | **Weekend** | Transaksi akhir pekan atau hari kerja |
            
            ---
            
            ### Tips
            
            - Jumlah transaksi tinggi (>$500) dapat memicu peringatan fraud
            - Transaksi larut malam (setelah 22:00) berisiko lebih tinggi
            - Beberapa kategori seperti belanja online memiliki tingkat fraud lebih tinggi
            """)
        
        st.markdown("---")
        
        # Show prediction history if exists
        if st.session_state.prediction_history:
            st.markdown("### Prediksi Terakhir")
            history_df = pd.DataFrame(st.session_state.prediction_history[-5:])  # Last 5
            st.dataframe(history_df, width='stretch')

