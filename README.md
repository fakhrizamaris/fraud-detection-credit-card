# 🛡️ Fraud Detection System

Sistem deteksi fraud untuk transaksi kartu kredit menggunakan **Machine Learning (Random Forest)** dengan interface web yang user-friendly.

---

## 📋 Daftar Isi

- [Tentang Project](#tentang-project)
- [Fitur](#fitur)
- [Struktur Project](#struktur-project)
- [Prerequisites](#prerequisites)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Penggunaan](#penggunaan)
- [Model Performance](#model-performance)
- [Presentation Notes](#presentation-notes)
- [Troubleshooting](#troubleshooting)

---

## 📖 Tentang Project

Project ini membangun sistem deteksi fraud untuk transaksi kartu kredit dengan:

- **Algoritma**: Random Forest Classifier
- **Dataset**: 14.000 transaksi (balanced: 50% fraud, 50% not fraud)
- **Akurasi Target**: Minimal 85%
- **Recall Target**: Minimal 80% (prioritas deteksi fraud)

---

## ✨ Fitur

### Feature Engineering

- `age` - Umur pemegang kartu (dari kolom `dob`)
- `hour` - Jam transaksi (0-23)
- `is_weekend` - Penanda transaksi akhir pekan
- `amt_per_hour_ratio` - Rasio jumlah transaksi per jam

### Model Evaluation Metrics

- ✅ Accuracy
- ✅ Precision
- ✅ Recall (PRIORITY)
- ✅ F1-Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix

### Web Dashboard (Streamlit)

- 📝 Form input untuk detail transaksi
- 📊 Hasil prediksi dengan confidence level
- 🎯 Analisis faktor risiko
- 💡 Rekomendasi tindakan
- 📈 Model Performance Dashboard
- 📥 Download hasil prediksi

---

## 📁 Struktur Project

```
fraud-detection/
│  # Streamlit dashboard
├── app.py
│  # Script training model
├── fraud_detection_rf.py
│  # Dependencies
├── requirements.txt
│  # Dataset
├── data/
│   └── credit_card_transactions2.csv
│ # Trained model + preprocessors
├── models/
│   └── fraud_detection_model.pkl
│ # notebook training
└── notebook/
    └── Fraud_detection_RF.ipynb
```

---

## 💻 Prerequisites

Pastikan Anda sudah menginstall:

- **Python** 3.10 - 3.12 (Project ini dikembangkan dengan Python 3.12)
- **pip** (Python package manager)
- **Virtual Environment** (opsional tapi direkomendasikan)

Untuk mengecek versi Python:

```bash
python --version
```

---

## 🚀 Instalasi

### Langkah 1: Clone/Download Project

Download project ini atau extract dari file ZIP.

### Langkah 2: Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
# Windows
python -m venv .venv

# jika sudah install uv
uv venv .venv # uv menginstall lebih cepat
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Langkah 3: Install Dependencies

```bash
pip install -r requirements.txt

# jika sudah install uv
uv pip install -r requirements.txt
```

### Langkah 4: Verifikasi Instalasi

```bash
pip list
```

Pastikan semua package terinstall dengan benar.

---

## ▶️ Cara Menjalankan

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

### Zip dan Ekstrak Project

#### 🗜️ Compress Project ke ZIP

Untuk membuat file ZIP dari project ini (misalnya untuk backup atau upload ke Colab), gunakan script yang sudah tersedia:

**Di Terminal/Command Prompt:**

```bash
python zip-project.py
```

**Di Jupyter Notebook:**

```python
# Jalankan di cell Jupyter
%run zip-project.py
```

Script akan membuat file `fraud-detection-project.zip` yang berisi seluruh project (kecuali folder `.venv`, `__pycache__`, dll).

#### 📦 Extract Project dari ZIP

Untuk extract file ZIP project:

**Di Terminal/Command Prompt:**

```bash
python unzip-project.py
```

**Di Jupyter Notebook:**

```python
# Jalankan di cell Jupyter
%run unzip-project.py
```

Script akan extract file ZIP ke folder yang sama dengan struktur aslinya.

---

### Upload Project ke Google Colab / Jupyter Server

Jika Anda ingin menjalankan project ini di **Google Colab** atau **Jupyter Server online**, ikuti langkah berikut:

#### Metode 1: Upload ZIP Manual

1. **Compress project** menggunakan `zip-project.py` (lihat bagian di atas)
2. Upload file `fraud-detection-project.zip` ke Google Colab
3. Extract menggunakan `unzip-project.py` di cell Colab:

```python
# Di Google Colab, upload file ZIP terlebih dahulu
from google.colab import files
uploaded = files.upload()  # Pilih file fraud-detection-project.zip

# Kemudian extract menggunakan script
%run unzip-project.py

# Pindah ke folder project
%cd fraud-detection

# Install dependencies
!pip install -r requirements.txt
```

---

## 📝 Penggunaan

### Input Data Transaksi

1. Buka dashboard Streamlit
   ![alt text](image.png)
2. Isi form di sidebar:

   - **Kategori Transaksi**: Jenis merchant (grocery_pos, gas_transport, dll)
   - **Jumlah Transaksi**: Nilai dalam USD ($0.01 - $100,000)
   - **Jenis Kelamin**: M (Male) / F (Female)
   - **Negara Bagian**: Lokasi transaksi (TX, NY, CA, dll)
   - **Umur**: 18-100 tahun
   - **Jam Transaksi**: 0-23 (format 24 jam)
   - **Akhir Pekan**: Centang jika Sabtu/Minggu

     ![alt text](image-1.png)

3. Klik tombol **"🔍 ANALISIS TRANSAKSI"**

### Membaca Hasil

![alt text](image-2.png)

- **✅ TRANSAKSI AMAN** (hijau): Transaksi tidak menunjukkan pola mencurigakan
- **🚨 POTENSI FRAUD** (merah): Transaksi terindikasi fraud, perlu verifikasi

---

## 📊 Model Performance

![alt text](image-3.png)

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

## 📝 Presentation Notes

Untuk catatan lengkap presentasi (penjelasan performa, fitur penting, dan FAQ), silakan baca:

**[📄 PRESENTATION_NOTES.md](PRESENTATION_NOTES.md)**

File tersebut berisi:

- Penjelasan detail metrik model
- Alasan mengapa akurasi tinggi (dataset balanced vs real-world)
- Pipeline training step-by-step
- Key features yang menjadi predictor kuat

---

## 🔧 Troubleshooting

### 1. Error: "Model belum di-training!"

**Penyebab:** File model tidak ditemukan di `models/fraud_detection_model.pkl`

**Solusi:**

```bash
python training_model.py
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

### 4. Streamlit tidak terbu\*\* di browser

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

## 📊 Contoh Test Case

### Test Case 1: Transaksi Aman

- Category: `grocery_pos`
- Amount: $50
- Gender: F
- State: TX
- Age: 35
- Hour: 14
- Weekend: No

**Expected Result:** ✅ TRANSAKSI AMAN (confidence > 80%)

### Test Case 2: Transaksi Mencurigakan

- Category: `gas_transport`
- Amount: $1,500
- Gender: M
- State: NY
- Age: 25
- Hour: 3
- Weekend: Yes

**Expected Result:** 🚨 POTENSI FRAUD (confidence > 70%)
