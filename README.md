# Credit Card Fraud Analysis

Sistem deteksi fraud untuk transaksi kartu kredit menggunakan **Machine Learning (Random Forest)** dengan interface web interaktif berbasis Streamlit.

---

## Daftar Isi

- [Tentang Project](#tentang-project)
- [Fitur Aplikasi](#fitur-aplikasi)
- [Struktur Project](#struktur-project)
- [Prerequisites](#prerequisites)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Penggunaan](#penggunaan)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)

---

## Tentang Project

Project ini membangun sistem deteksi fraud untuk transaksi kartu kredit dengan:

- **Algoritma**: Random Forest Classifier
- **Dataset**: 14.000 transaksi (balanced: 50% fraud, 50% not fraud)
- **Akurasi Target**: Minimal 85%
- **Recall Target**: Minimal 80% (prioritas deteksi fraud)
- **Bahasa Interface**: Indonesia (dengan istilah teknis dalam bahasa Inggris)

---

## Fitur Aplikasi

### Tab-Tab Dashboard

| Tab                   | Deskripsi                                                           |
| --------------------- | ------------------------------------------------------------------- |
| **About Dataset**     | Informasi latar belakang, dataset, metodologi, dan manfaat sistem   |
| **Dashboard**         | Eksplorasi data historis dengan visualisasi EDA lengkap             |
| **Fraud Detection**   | Form input transaksi dan hasil analisis prediksi fraud              |
| **Machine Learning**  | Penjelasan pipeline training model (preprocessing, SMOTE, training) |
| **Model Performance** | Dashboard evaluasi performa model dan feature importance            |
| **Contact Me**        | Informasi profil pengembang dan kontak                              |

### Feature Engineering

- `age` - Umur pemegang kartu (dari kolom `dob`)
- `hour` - Jam transaksi (0-23)
- `is_weekend` - Penanda transaksi akhir pekan
- `amt_per_hour_ratio` - Rasio jumlah transaksi per jam

### Model Evaluation Metrics

- Accuracy
- Precision
- Recall (PRIORITY)
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### Fitur Dashboard

- Form input untuk detail transaksi
- Hasil prediksi dengan confidence level
- Kategori risiko (Amount, Time, Age, Day Type)
- Analisis faktor risiko
- Rekomendasi tindakan
- Download hasil prediksi (CSV)
- Visualisasi EDA interaktif
- Correlation heatmap

---

## Struktur Project

```
fraud-detection/
├── app.py                    # Main Streamlit application
├── fraud_detection_rf.py     # Script training model
├── requirements.txt          # Python dependencies
├── README.md                 # Dokumentasi ini
│
├── data/
│   └── credit_card_transactions2.csv    # Dataset
│
├── models/
│   └── fraud_detection_model.pkl        # Trained model + preprocessors
│
├── tabs/                     # Modul tab Streamlit
│   ├── about_dataset.py      # Tab About Dataset
│   ├── dashboard.py          # Tab Dashboard (EDA)
│   ├── fraud_detection.py    # Tab Fraud Detection
│   ├── machine_learning.py   # Tab Machine Learning
│   ├── model_performance.py  # Tab Model Performance
│   └── contact_me.py         # Tab Contact Me
│
└── notebook/
    └── Fraud_detection_RF.ipynb    # Jupyter Notebook training
```

---

## Prerequisites

Pastikan Anda sudah menginstall:

- **Python** 3.10 - 3.12 (Project ini dikembangkan dengan Python 3.12)
- **pip** (Python package manager)
- **Virtual Environment** (opsional tapi direkomendasikan)

Untuk mengecek versi Python:

```bash
python --version
```

---

## Instalasi

### Langkah 1: Clone/Download Project

Download project ini atau extract dari file ZIP.

### Langkah 2: Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Langkah 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Langkah 4: Verifikasi Instalasi

```bash
pip list
```

Pastikan semua package terinstall dengan benar.

---

## Cara Menjalankan

### Menjalankan Training Model (Opsional)

Jika ingin re-training model:

```bash
python fraud_detection_rf.py
```

**Output yang diharapkan:**

```
✅ Total data: 14000 rows
✅ Features setelah engineering: ['category', 'amt', 'gender', ...]
📊 Train: 11200 | Test: 2800
🎯 HASIL EVALUASI MODEL
Accuracy  : 0.92xx (92.xx%)
Precision : 0.91xx (91.xx%)
Recall    : 0.93xx (93.xx%)
...
✅ Model berhasil disimpan ke 'models/fraud_detection_model.pkl'
```

### Menjalankan Streamlit Dashboard

```bash
streamlit run app.py
```

**Browser akan otomatis terbuka di:** `http://localhost:8501`

### Menjalankan Jupyter Notebook

```bash
jupyter notebook notebook/Fraud_detection_RF.ipynb
```

Atau buka melalui VS Code/JupyterLab.

---

## Penggunaan

### Tab About Dataset

Berisi informasi tentang:

- Latar belakang fraud kartu kredit
- Fitur-fitur dalam dataset
- Metodologi machine learning yang digunakan
- Manfaat sistem deteksi fraud

### Tab Dashboard

Eksplorasi data historis dengan visualisasi lengkap:

1. Overview Dataset (jumlah transaksi, fraud rate, dll)
2. Pemeriksaan Kualitas Data (missing values, data types)
3. Deteksi Outlier (box plot Amount dan Age)
4. Normalisasi Data (perbandingan sebelum/sesudah)
5. EDA - Distribusi Kategorikal (gender, kategori, jam, weekend)
6. EDA - Distribusi Numerik (usia, amount)
7. Analisis Pola Fraud (fraud per kategori, fraud per jam)
8. Analisis Korelasi (heatmap)

### Tab Fraud Detection

1. Isi form di sidebar:

   - **Kategori Transaksi**: Jenis merchant (grocery_pos, gas_transport, dll)
   - **Jumlah Transaksi**: Nilai dalam USD ($0.01 - $100,000)
   - **Jenis Kelamin**: Laki-laki / Perempuan
   - **Negara Bagian**: Lokasi transaksi (TX, NY, CA, dll)
   - **Usia**: 18-100 tahun
   - **Jam Transaksi**: 0-23 (format 24 jam)
   - **Akhir Pekan**: Centang jika Sabtu/Minggu

2. Klik tombol **"ANALISIS TRANSAKSI"**

3. Lihat hasil:
   - **Kategori Risiko**: Amount Risk, Time Risk, Age Group, Day Type
   - **Hasil Prediksi**: TRANSAKSI AMAN atau POTENSI FRAUD TERDETEKSI
   - **Distribusi Probabilitas**: Pie chart dan detail
   - **Faktor Analisis**: Faktor yang mempengaruhi hasil
   - **Unduh Hasil**: Download CSV

### Tab Machine Learning

Penjelasan pipeline training model:

1. Data Preprocessing (missing values, feature engineering, encoding)
2. Penanganan Class Imbalance (SMOTE)
3. Training Model (Random Forest hyperparameters)
4. Feature Importance

### Tab Model Performance

Dashboard evaluasi model:

- Informasi Model (algoritma, n_estimators, max_depth)
- Metrik Performa (Accuracy, Recall, Precision, F1-Score, ROC-AUC)
- Feature Importance (bar chart dan tabel)
- Riwayat Prediksi (history dengan download)

### Tab Contact Me

Informasi profil pengembang:

- Data penulis
- Detail kontak
- Latar belakang akademik
- Minat penelitian
- Tentang proyek
- Ucapan terima kasih

---

## Model Performance

Model Random Forest yang digunakan telah mencapai performa tinggi dengan metrik berikut:

| Metrik        | Score  | Status          |
| ------------- | ------ | --------------- |
| **Accuracy**  | ~97.2% | ✅ Target: ≥85% |
| **Recall**    | ~97.4% | ✅ Target: ≥80% |
| **Precision** | ~97.0% | ✅ Target: ≥75% |
| **F1-Score**  | ~97.2% | ✅ Excellent    |
| **ROC-AUC**   | ~0.997 | ✅ Outstanding  |

**Catatan Penting:**

- Dataset yang digunakan sudah pre-balanced (50% fraud, 50% safe)
- Di real-world, fraud rate biasanya < 1%, sehingga performa akan berbeda
- Cross-validation menunjukkan model stabil dan tidak overfitting

**Top 3 Fitur Paling Berpengaruh:**

1. `amt` (Transaction Amount) - 45.5%
2. `amt_per_hour_ratio` - 22.9%
3. `hour` (Transaction Hour) - 18.5%

---

## Troubleshooting

### 1. Error: "Model belum di-training!"

**Penyebab:** File model tidak ditemukan di `models/fraud_detection_model.pkl`

**Solusi:**

```bash
python fraud_detection_rf.py
```

### 2. Error: ModuleNotFoundError

**Penyebab:** Dependencies belum terinstall

**Solusi:**

```bash
pip install -r requirements.txt
```

### 3. Error: "No module named 'sklearn'"

**Penyebab:** scikit-learn belum terinstall

**Solusi:**

```bash
pip install scikit-learn==1.3.0
```

### 4. Streamlit tidak terbuka di browser

**Penyebab:** Port 8501 sudah digunakan

**Solusi:**

```bash
streamlit run app.py --server.port 8502
```

### 5. Jupyter Notebook tidak bisa run

**Penyebab:** Kernel tidak terdeteksi

**Solusi:**

```bash
python -m ipykernel install --user --name=fraud-detection
```

### 6. Error: "Permission denied" saat install

**Penyebab:** Tidak punya akses admin

**Solusi:**

```bash
pip install --user -r requirements.txt
```

---

## Contoh Test Case

### Test Case 1: Transaksi Aman

- Category: `grocery_pos`
- Amount: $50
- Gender: Perempuan
- State: TX
- Age: 35
- Hour: 14
- Weekend: No

**Expected Result:** TRANSAKSI AMAN (confidence > 80%)

### Test Case 2: Transaksi Mencurigakan

- Category: `gas_transport`
- Amount: $1,500
- Gender: Laki-laki
- State: NY
- Age: 25
- Hour: 3
- Weekend: Yes

**Expected Result:** POTENSI FRAUD TERDETEKSI (confidence > 70%)

---

## Teknologi yang Digunakan

- **Python 3.12**
- **Streamlit** - Web Framework
- **Scikit-learn** - Machine Learning
- **Pandas & NumPy** - Data Processing
- **Altair & Matplotlib** - Visualization
- **Imbalanced-learn** - SMOTE for class balancing

---

_Credit Card Fraud Analysis System - Dikembangkan untuk tujuan akademik_
