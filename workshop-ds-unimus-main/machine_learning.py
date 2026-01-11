import streamlit as st
import pandas as pd
import numpy as np
import math
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import ConfusionMatrixDisplay


def ml_model():
    df = pd.read_excel('healthcare-dataset-stroke-data.xlsx')
    df = df.drop(columns=['id'])

    #1. Membagi kolom numerik dan kategorik
    numbers = df.drop(columns=['hypertension', 'heart_disease','stroke']).select_dtypes(include=['number']).columns
    categories = df.select_dtypes(exclude=['number']).columns

    #2. Deteksi dan penanganan outlier dengan IQR Method
    st.write('### 1. Deteksi Outlier') 
    Q1 = df[numbers].quantile(0.25)
    Q3 = df[numbers].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    st.write(f"Jumlah data sebelum pembersihan: **{df.shape[0]} baris**")
    df = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]
    st.write(f"Jumlah data setelah pembersihan outlier: **{df.shape[0]} baris**")

    #3. Membaca Dataset
    df_select = df.copy()
    st.write('**Dataset yang digunakan**')
    st.dataframe(df.head())

    #4. Memisahkan variabel numerik dan kategorik bagian 2
    numbers = df_select.drop(columns=['stroke', 'hypertension',
                                  'heart_disease', 'chronic_count',
                                  'risk_score']).select_dtypes(include=['number']).columns
    categories = df_select.select_dtypes(exclude=['number']).columns

    #5. Feature Encoding
    df_select = pd.get_dummies(df_select, drop_first=True)

    #6. Normalisasi kolom numerik dengan MinMax Scaler
    st.write('### 2. Normalisasi menggunakan MinMax Scaler')
    for col in numbers:
        df_select[col] = MinMaxScaler().fit_transform(df_select[col].values.reshape(len(df_select), 1))

    col1, col2 = st.columns(2)
    with col1:
    #6.a Visualisasi dengan Density Plot
        st.write('**Sebelum Normalisasi**')
        for col in numbers:
            chart = []
            chart = (
                alt.Chart(df).transform_density(col,as_=[col, 'density'])
                .mark_area(opacity=0.5)
                .encode(x=f"{col}:Q",y="density:Q")
                .properties(width=350,height=200,title=f"Density Plot: {col}")
            )
    st.altair_chart(chart, use_container_width=True)

    with col2:
    #6.b Visualisasi dengan Density Plot
        st.write('**Setelah Normalisasi**')
        for col in numbers:
            chart = (alt.Chart(df_select).transform_density(col,as_=[col, 'density'])
                .mark_area(opacity=0.5)
                .encode(x=f"{col}:Q",y="density:Q")
                .properties(width=350,height=200,title=f"Density Plot: {col}")
            )
    st.altair_chart(chart, use_container_width=True) 
    
    #7. Correlation Heatmap untuk melihat korelasi linear antara kolom-kolom numerik
    st.write('### 3. Korelasi Linear antar Kolom Numerik')
    col1, col2 = st.columns(2)
    with col1:
    #7.a Correlation Heatmap
        st.write('**Correlation Heatmap**')
        corr = df[numbers].corr().reset_index().melt('index')
        corr.columns = ['Variable1', 'Variable2', 'Correlation']
        heatmap = (
            alt.Chart(corr).mark_rect()
            .encode(
                x=alt.X('Variable2:N', title=None),
                y=alt.Y('Variable1:N', title=None),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blues')),
                tooltip=['Variable1', 'Variable2', alt.Tooltip('Correlation:Q', format='.2f')]
            )
            .properties(width=400,height=400,title="Correlation Heatmap (Altair)"))

        # Tambahkan nilai angka di dalam kotak
        text = (alt.Chart(corr).mark_text(fontSize=12, color='black')
            .encode(x='Variable2:N',y='Variable1:N',text=alt.Text('Correlation:Q', format=".2f")))
        st.altair_chart(heatmap + text, use_container_width=True)
    
    with col2:
    #7.b Deskripsi Correlation Heatmap
        st.write('**Deskripsi Correlation Heatmap**')
        st.write("""
        - Korelasi **age–bmi = 0.30**, Korelasi positif yang lemah hingga sedang. 
            Peningkatan usia berhubungan dengan sedikit peningkatan BMI.
        - Korelasi **age–avg_glucose_level = 0.17**, Korelasi positif yang sangat lemah. 
            Peningkatan usia sedikit berhubungan dengan peningkatan rata-rata tingkat glukosa.
        - Korelasi **avg_glucose_level–bmi = 0.17**, Korelasi positif yang sangat lemah. 
            Peningkatan rata-rata tingkat glukosa sedikit berhubungan dengan peningkatan BMI.
        - Tidak ada korelasi kuat antar variabel, sehingga risiko multikolinearitas rendah.
        """)
    
    #8. Train Test Split
    st.write("### 4. Train–Test Split ")
    X = df_select.drop("stroke", axis=1)
    y = df_select["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"X_train: {len(X_train)}", f", X_test: {len(X_test)}")
    
    #9. Handling Imbalance Class dengan SMOTE
    st.write("### 5. Handling Imbalance Class")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **Mengapa Imbalance Class Penting?**
        1. Ketika Label 1 sangat sedikit, model lebih sering menebak Label 0 untuk mendapatkan akurasi tinggi.
        2. Pada kasus medis seperti stroke, mendeteksi kelas 1 jauh lebih penting.
        3. Model hanya belajar dari pola kelas mayoritas sehingga performanya buruk pada data riil.
        """)
    with col2:
        st.write('**Sebelum Balancing Class**')
        col1, col2 = st.columns(2)
        with col1:
            total_0 = (y_train == 0).sum()
            st.metric(label="Label 0", value = total_0)
        with col2:
            total_1 = (y_train == 1).sum()
            st.metric(label="Label 1", value = total_1)
        
        st.write('**Setelah Balancing Class**')
        sm = SMOTE(random_state=42)
        X_train_balance, y_train_balance = sm.fit_resample(X_train, y_train)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Label 0", (y_train_balance == 0).sum())
        with col2:
            st.metric("Label 1", (y_train_balance == 1).sum())
    
    #9. Training Model dengan Logistic Regression
    st.write('### 6. Pemodelan')
    model = LogisticRegression()
    model.fit(X_train_balance, y_train_balance)
    train_accuracy = model.score(X_train_balance, y_train_balance)
    st.write("Akurasi Training =", round(train_accuracy * 100, 2), "%")

    col1, col2 = st.columns([6,4])
    with col1:
        st.write('**Parameter Model Logistic Regression**')
        feature_names = X_train_balance.columns
        beta_0 = model.intercept_[0]           # Intercept
        beta = model.coef_[0]                  # Koefisien fitur

        st.write("**β0 (Intercept)**")
        st.write(beta_0)

        st.write("**β1, β2, ..., βn (Koefisien per Feature)**")
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient (β)": beta}).sort_values(by="Coefficient (β)", ascending=False)    
        st.dataframe(coef_df)
    
    with col2:
        st.write('**Interpretasi Model**')
        st.write("**a. Koefisien Positif → Meningkatkan Risiko Stroke**")
        st.write("Semakin besar nilai β, semakin tinggi pengaruh feature " \
                    "tersebut terhadap kenaikan peluang stroke.")
        
        st.write("**b. Koefisien Negatif → Menurunkan Risiko Stroke**")
        st.write("Semakin kecil (negatif) nilai β, semakin tinggi pengaruh feature "
                    "tersebut terhadap penurunan peluang stroke.")
        
        st.write('**c. Intercept Model (β0)**')
        st.markdown(f"""
            Nilai β0 = **{beta_0:.4f}**  
            menunjukkan bahwa peluang dasar (ketika semua variabel independen 
            tidak ada atau diatur ke nol) untuk terkena stroke sangat rendah..""")
        
    st.write('**d. Faktor Risiko Tertinggi**')
    st.markdown(f"""
        Berdasarkan koefisien yang besar:
        - **Usia.** Peningkatan usia memiliki hubungan paling besar dalam meningkatkan peluang terkena stroke.
        - **bmi_category_3. Overweight.** Memiliki status Overweight (Kelebihan Berat Badan) secara signifikan 
                meningkatkan peluang terkena stroke.
        - **bmi_category_2. Normal.** Menariknya, memiliki status Normal juga meningkatkan terkena stroke, 
                meskipun lebih rendah daripada Overweight. Ini mungkin karena kategori ini adalah perbandingan 
                dengan kategori dasar (baseline) yang tidak tercantum (misalnya, Underweight).
        - **age_category_3. Adult.** Berada dalam kategori usia dewasa meningkatkan peluang terkena stroke.
        - **bmi_category_4. Obesity.** Meskipun merupakan kategori risiko tinggi, status Obesity memiliki 
                koefisien yang sedikit lebih rendah daripada Overweight dan Normal. Hal ini dapat mengindikasikan 
                adanya interaksi atau bahwa data Obesity lebih terbatas atau kurang signifikan dalam model ini.""")
        
    st.write('**e. Insight Tambahan**')
    st.markdown("""
        - **Usia adalah Faktor Kunci (Age is King):** Variabel usia (age) kontinu adalah prediktor tunggal yang paling 
                berpengaruh dalam model ini, jauh melebihi variabel kategori lainnya.
        - **BMI Signifikan:** Kategori BMI memiliki pengaruh signifikan dalam meningkatkan seseorang berpeluang terkena stroke.
        - **Prioritas Risiko:** Secara umum, model mengidentifikasi bahwa usia dan status berat badan adalah faktor utama 
                yang mendorong kemungkinan seseorang mengalami stroke.
        """) 
    
    #10 Evaluasi Model
    st.write('### 7. Evaluasi Model')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision= precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    col1, col2 = st.columns([2,2])
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        labels = model.classes_
        cm_df = pd.DataFrame(cm, columns=labels, index=labels).reset_index().melt(id_vars='index')
        cm_df.columns = ['Actual', 'Predicted', 'Count']
        chart = (alt.Chart(cm_df).mark_rect()
            .encode(
                x=alt.X("Predicted:N", title="Predicted"),
                y=alt.Y("Actual:N", title="Actual"),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Actual", "Predicted", "Count"]
            )
        )
        text = (
            alt.Chart(cm_df).mark_text(color="black")
            .encode(x="Predicted:N",y="Actual:N",text="Count:Q"))
        st.altair_chart(chart + text, use_container_width=True)
        
    with col2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Akurasi", value=f"{round(accuracy * 100, 2)}%")
            st.metric(label="Precision", value=f"{round(precision * 100, 2)}%")
            st.metric(label="Recall", value=f"{round(recall * 100, 2)}%")
        with col2:
            st.metric(label="F1 Score", value=f"{round(f1 * 100, 2)}%")
            st.metric(label="ROC AUC", value=f"{round(roc_auc * 100, 2)}%")

    #11. Insight dari Hasil Pemodelan
    st.write('### 8. Evaluasi dari Hasil Pemodelan')    
    col1, col2 = st.columns(2)
    with col1:
        st.write('**Kualitas Prediksi Keseluruhan**')
        st.markdown(f"""
            Berdasarkan hasil dari model tersebut, dapat disimpulkan bahwa model:
            - **Akurasi.** Sekitar 76% dari total prediksi model sudah benar 
                    (baik memprediksi Stroke maupun Tidak Stroke).
            - **ROC AUC.** Nilai AUC yang sangat baik **(86,08%)** menunjukkan kemampuan model 
                    untuk membedakan antara pasien Stroke dan Tidak Stroke. 
                    Model memiliki kinerja diskriminasi yang kuat.
            - **F1 Score.** F1 Score yang cukup baik **(71,3%)**, menunjukkan keseimbangan 
                    antara Precision dan Recall.""")
        st.write('**Fokus pada Kelas Positif (Stroke = 1)**')
        st.write("Dalam memprediksi Stroke, kita harus meminimalkan **False Negatives (FN)**," \
                    " yaitu pasien yang sebenarnya Stroke tetapi diprediksi Tidak Stroke.")
        st.markdown(f"""
            Berdasarkan hasil dari model tersebut, dapat disimpulkan bahwa model:
            - **Recall (Sensitivitas): 88.17%.** Dari semua pasien yang benar-benar menderita Stroke (True 1), 
                    model berhasil mengidentifikasi 88.17% di antaranya. 
            - **Kualitas Recall Sangat Baik.** Mengingat konsekuensi serius dari Stroke yang tidak terdiagnosis, 
                    Recall yang tinggi sangat penting untuk meminimalkan risiko medis yang terlewatkan (FN = 31).
            - **Precision: 59.84%.** Dari semua pasien yang diprediksi model sebagai Stroke (Predicted 1), 
                    hanya 59.84% yang benar-benar menderita Stroke (sisanya adalah False Positives = 155).
            - **Kualitas: Cukup Rendah.** Model Anda sering menghasilkan alarm palsu (False Positives). 
                    Ini berarti 155 pasien yang tidak menderita Stroke harus menjalani pengujian atau 
                    kekhawatiran yang tidak perlu.""")
        
    with col2:
        st.write('**Rekomendasi Peningkatan Kualitas Model (Metrik)**')
        st.write('**a. Tingkatkan Threshold (Batas Ambang):** Saat ini, model kemungkinan menggunakan threshold ' \
                    'probabilitas default (biasanya 0.5). Untuk mengurangi False Positives (FP), dapat meningkatkan ' \
                    'threshold (misalnya, menjadi 0.6 atau 0.7). Ini akan membuat model lebih "skeptis" dan hanya ' \
                    'memprediksi Stroke jika probabilitasnya sangat tinggi.')
        st.write('**b. Coba Model Lain:** Model seperti Random Forest atau Gradient Boosting mungkin lebih baik dalam ' \
                    'menangani hubungan kompleks dan dapat meningkatkan Precision tanpa mengorbankan Recall secara drastis.')
        st.write('**Rekomendasi Terhadap Masyarakat Umum**')
        st.write("""
        - Fokus pada pencegahan stroke di kelompok usia dewasa dan lanjut
        - Kampanye berhenti merokok, terutama bagi yang memiliki riwayat merokok
        - Promosi gaya hidup sehat untuk mengendalikan berat badan dan mencegah obesitas
        - Perhatian khusus pada pekerja dengan tingkat stres tinggi
        - Deteksi dini dan intervensi pada individu berisiko tinggi berdasarkan faktor-faktor di atas
        """)
    
    import joblib

    # Simpan model dan fitur kolom agar prediction.py bisa pakai
    joblib.dump(model, "model_stroke.pkl")
    joblib.dump(feature_names, "model_features.pkl")
    joblib.dump(numbers, "numeric_columns.pkl")
