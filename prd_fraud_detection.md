# Product Requirements Document (PRD)
## Fraud Detection System - Machine Learning Project

---

### Document Information
| Item | Detail |
|------|--------|
| **Project Name** | Credit Card Fraud Detection System |
| **Version** | 1.0 |
| **Date** | 10 Januari 2026 |
| **Client** | [Nama Client] |
| **Engineer** | [Nama Anda] |
| **Budget** | Rp 150.000 |
| **Deadline** | 11 Januari 2026, 06:00 WIB |

---

## 1. PROJECT OVERVIEW

### 1.1 Tujuan Project
Membangun sistem deteksi fraud untuk transaksi kartu kredit menggunakan Machine Learning dengan interface web yang user-friendly untuk keperluan tugas kuliah.

### 1.2 Success Criteria
- Model accuracy minimal 85%
- Recall rate minimal 80% (prioritas deteksi fraud)
- Web dashboard dapat running tanpa error
- Semua deliverables lengkap sesuai checklist
- Code mudah dipahami untuk presentasi ke dosen

---

## 2. SCOPE DETAIL

### 2.1 IN SCOPE (Yang Dikerjakan)

#### A. Data Processing & Feature Engineering
**Input:**
- Dataset: `credit_card_transactions2.csv` (14,000 rows)
- Status: Already balanced (50% fraud, 50% not fraud)

**Process:**
- [x] Load dan validasi dataset
- [x] Create feature `age` dari kolom `dob`
- [x] Extract feature `hour` dari `trans_date_trans_time`
- [x] Create feature `is_weekend` (pattern detection)
- [x] Create feature `amt_per_hour_ratio` (anomaly detection)
- [x] Drop irrelevant columns: `Unnamed: 0`, `cc_num`, `first`, `last`, `street`, `trans_num`, `unix_time`, `trans_date_trans_time`, `dob`, `merchant`, `job`, `zip`, `lat`, `long`, `merch_lat`, `merch_long`, `merch_zipcode`, `city_pop`, `city`

**Output Features:**
- Categorical: `category`, `gender`, `state`
- Numerical: `amt`, `age`, `hour`, `is_weekend`, `amt_per_hour_ratio`
- Target: `is_fraud`

#### B. Model Training
**Algorithm:**
- Primary: Random Forest

**Configuration:**
- Train-test split: 80/20
- Stratified sampling (maintain balance)
- Random state: 42 (reproducibility)

**Preprocessing:**
- Label Encoding untuk categorical features
- StandardScaler untuk numerical features

**Evaluation Metrics:**
- [x] Accuracy
- [x] Precision
- [x] Recall (PRIORITY metric)
- [x] F1-Score
- [x] ROC-AUC Score
- [x] Confusion Matrix

**Visualization:**
- [x] Confusion Matrix heatmap
- [x] Feature Importance chart

#### C. Streamlit Web Dashboard

**UI Components:**

**Header Section:**
- Title: "Fraud Detection System"
- Subtitle: "Early Warning System untuk Deteksi Transaksi Mencurigakan"

**Sidebar (Input Form):**
- [x] Dropdown: Category (pilihan dari data training)
- [x] Number input: Amount (min: 0.01, max: 100,000)
- [x] Dropdown: Gender (M/F)
- [x] Dropdown: State (pilihan dari data training)
- [x] Slider: Age (18-100)
- [x] Slider: Hour (0-23)
- [x] Checkbox: Is Weekend
- [x] Button: "ANALISIS TRANSAKSI" (primary button)

**Main Section (Output):**
- [x] Metrics display: Amount, Hour, Location
- [x] Prediction result (color-coded):
  - Green (success): "TRANSAKSI AMAN"
  - Red (error): "POTENSI INDIKASI FRAUD TERDETEKSI!"
- [x] Confidence percentage
- [x] Probability breakdown (Aman vs Fraud)
- [x] Progress bar visualization
- [x] Risk factors analysis (conditional)
- [x] Recommended actions (untuk fraud case)

**Footer:**
- System version info

#### D. Deliverables

**Code Files:**
1. [x] `training_model.py`
   - Fully commented (Indonesian)
   - Print statements untuk tracking progress
   - Error handling
   - Model saving mechanism

2. [x] `app.py`
   - Clean code structure
   - User-friendly interface
   - Error handling untuk missing model
   - Responsive layout

3. [x] `fraud_detection_model.pkl`
   - Trained model
   - Scaler object
   - Label encoders
   - Feature metadata

**Documentation Files:**
4. [x] `requirements.txt`
   - All dependencies dengan version
   - Format: `package==version`

5. [x] `README.md`
   - Project description
   - Installation steps (step-by-step)
   - Usage instructions
   - Troubleshooting guide
   - Example screenshots (optional)

**Bonus Files:**
6. [x] Google Colab notebook version
   - Cell-by-cell execution
   - Inline visualizations
   - Download mechanism untuk model

#### E. Documentation & Support

**Documentation:**
- [x] Code comments dalam Bahasa Indonesia
- [x] Inline explanations untuk algoritma penting
- [x] Docstrings untuk fungsi utama

**Support:**
- [x] Installation guide (step-by-step)
- [x] Troubleshooting common errors
- [x] 1x free revision untuk bug fixes

---

### 2.2 OUT OF SCOPE (Yang TIDAK Dikerjakan)

#### Explicitly Excluded:
- [ ] Deep Learning models (Neural Networks, LSTM, dll)
- [ ] Real-time database integration
- [ ] User authentication/login system
- [ ] Production deployment (Heroku, AWS, GCP)
- [ ] API development (REST/GraphQL)
- [ ] Mobile app development
- [ ] Advanced hyperparameter tuning (GridSearch/RandomSearch)
- [ ] Model ensemble (stacking multiple models)
- [ ] Real-time data streaming
- [ ] Email/SMS notification system
- [ ] Payment gateway integration
- [ ] Multi-language support
- [ ] Advanced data visualization dashboard (Plotly Dash, Tableau)
- [ ] A/B testing framework
- [ ] Model monitoring & retraining pipeline
- [ ] Data augmentation techniques
- [ ] Custom dataset collection/scraping

#### Conditional Items (Need Client Confirmation):
- [ ] Online deployment (Streamlit Cloud) - **Not included by default**
- [ ] PowerPoint presentation - **Not included by default**
- [ ] Video tutorial - **Not included by default**
- [ ] Written report/paper - **Not included by default**

---

## 3. TECHNICAL SPECIFICATIONS

### 3.1 Technology Stack

**Programming Language:**
- Python 3.8 - 3.10 (compatible range)

**Core Libraries:**
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
streamlit==1.28.0
matplotlib==3.7.2
seaborn==0.12.2
```

**Development Environment:**
- Local: VS Code / PyCharm / Jupyter Notebook
- Cloud: Google Colab (backup)

### 3.2 Performance Requirements

**Model Performance:**
- Minimum Accuracy: 85%
- Minimum Recall: 80% (critical for fraud detection)
- Minimum Precision: 75%
- Training time: < 2 minutes (14k rows)

**Application Performance:**
- Prediction response time: < 1 second
- Dashboard load time: < 3 seconds
- No memory leaks or crashes

### 3.3 Code Quality Standards

**Mandatory:**
- No hardcoded paths (use relative paths)
- Proper error handling (try-except blocks)
- Consistent naming conventions (snake_case)
- Comments for complex logic
- No unused imports/variables

**Code Structure:**
```
project/
├── training_model.py       # Training script
├── app.py                  # Streamlit app
├── fraud_detection_model.pkl   # Saved model
├── requirements.txt        # Dependencies
├── README.md              # Documentation
└── data/
    └── credit_card_transactions2.csv
```

---

## 4. ACCEPTANCE CRITERIA

### 4.1 Training Script (training_model.py)

**Must Pass:**
- [x] Successfully loads dataset tanpa error
- [x] Creates all 4 engineered features correctly
- [x] Prints evaluation metrics ke console
- [x] Saves model file (.pkl) successfully
- [x] Model file size < 50MB
- [x] No warnings/errors during execution
- [x] Reproducible results (same random_state)

**Output Example:**
```
✅ Total data: 14000 rows
✅ Features setelah engineering: ['category', 'amt', 'gender', ...]
📊 Train: 11200 | Test: 2800
🎯 HASIL EVALUASI MODEL
Accuracy  : 0.9245 (92.45%)
Precision : 0.9102 (91.02%)
Recall    : 0.9389 (93.89%)
F1-Score  : 0.9244
✅ Model berhasil disimpan
```

### 4.2 Streamlit Dashboard (app.py)

**Must Pass:**
- [x] Loads model tanpa error
- [x] All input fields working correctly
- [x] Prediction button triggers analysis
- [x] Correct output untuk safe transaction (green)
- [x] Correct output untuk fraud transaction (red)
- [x] Confidence percentage displayed
- [x] No crashes dengan invalid input
- [x] UI responsive di browser modern

**Test Cases:**

**Test Case 1: Safe Transaction**
- Input: Category=grocery_pos, Amount=50, Gender=F, State=TX, Age=35, Hour=14, Weekend=No
- Expected: "TRANSAKSI AMAN" (green message)
- Confidence: > 80%

**Test Case 2: Suspicious Transaction**
- Input: Category=gas_transport, Amount=1500, Gender=M, State=NY, Age=25, Hour=3, Weekend=Yes
- Expected: "POTENSI INDIKASI FRAUD TERDETEKSI!" (red message)
- Confidence: > 70%

### 4.3 Documentation

**README.md Must Include:**
- [x] Project title & description
- [x] Prerequisites (Python version, pip)
- [x] Installation steps (numbered list)
- [x] How to run training script
- [x] How to run Streamlit app
- [x] Troubleshooting section (min 3 common issues)

**requirements.txt Must Include:**
- [x] All libraries dengan exact versions
- [x] No redundant packages
- [x] Tested & verified working

---

## 5. TIMELINE & MILESTONES

### Phase 1: Development (6 hours)
- **Hour 1-2:** Data processing & feature engineering
- **Hour 3-4:** Model training & evaluation
- **Hour 5-6:** Streamlit dashboard development

### Phase 2: Testing (1.5 hours)
- **Hour 7:** Test all functionalities
- **Hour 8:** Bug fixes & optimization

### Phase 3: Documentation (1 hour)
- **Hour 9:** Write README & comments

### Phase 4: Delivery (0.5 hour)
- **Hour 10:** Package all files, final check, delivery

**Total Estimated Time:** 8-10 hours
**Buffer Time:** 2 hours untuk unforeseen issues
**Hard Deadline:** 11 Januari 2026, 06:00 WIB

---

## 6. RISK MANAGEMENT

### Potential Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Dataset corrupt/missing columns | High | Low | Validate dataset structure first |
| Library compatibility issues | Medium | Medium | Use tested versions in requirements.txt |
| Model accuracy < 85% | High | Low | Use XGBoost with proven config |
| Streamlit deployment error | Medium | Low | Test locally before delivery |
| Client change requirement | High | Medium | **Refer to this PRD - out of scope = extra charge** |

---

## 7. CHANGE REQUEST POLICY

### Scope Creep Prevention

**Any request beyond this PRD requires:**
1. Written confirmation from client
2. Additional timeline extension
3. Additional budget negotiation

**Examples of Change Requests:**
- "Bisa tambahin fitur export PDF?"  → **Out of scope** (+50k, +2 hours)
- "Deploy ke cloud dong" → **Out of scope** (+30k, +1 hour)
- "Bikin presentasi PPT juga" → **Out of scope** (+40k, +2 hours)
- "Ganti algoritma ke Deep Learning" → **Out of scope** (+100k, +5 hours)

**Response Template:**
```
Halo Kak, request ini di luar scope PRD yang sudah disepakati.
Untuk menambahkan fitur [X], perlu:
- Tambahan waktu: [Y] jam
- Tambahan biaya: Rp [Z]
Apakah Kakak mau proceed dengan change request ini?
```

---

## 8. DELIVERY CHECKLIST

### Final Delivery Package

**Before Handover, Verify:**
- [ ] All 5 core files present (training_model.py, app.py, .pkl, requirements.txt, README.md)
- [ ] Google Colab notebook included
- [ ] Model file < 50MB
- [ ] README has clear instructions
- [ ] Tested on fresh Python environment
- [ ] No absolute paths in code
- [ ] All evaluation metrics printed correctly
- [ ] Streamlit runs without errors
- [ ] Test cases passed (safe & fraud scenarios)
- [ ] Comments in Indonesian
- [ ] No sensitive data hardcoded

**Delivery Method:**
- ZIP file via Discord/Email
- Include: "Project selesai sesuai PRD v1.0"

---

## 9. SIGN-OFF

### Engineer Commitment
Saya berkomitmen untuk menyelesaikan project sesuai scope PRD ini dengan kualitas terbaik dalam timeline yang disepakati.

**Signature:** ________________  
**Date:** 10 Januari 2026

### Client Acknowledgement
Saya memahami dan menyetujui scope, timeline, dan deliverables dalam PRD ini.

**Signature:** ________________  
**Date:** ________________

---

## 10. APPENDIX

### A. Glossary
- **Fraud Detection:** Sistem identifikasi transaksi mencurigakan
- **Recall:** Persentase fraud yang berhasil dideteksi
- **Precision:** Persentase prediksi fraud yang benar
- **XGBoost:** Algoritma Gradient Boosting yang powerful
- **Streamlit:** Framework Python untuk web dashboard

### B. Reference Links
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Streamlit Documentation: https://docs.streamlit.io/
- Scikit-learn Guide: https://scikit-learn.org/stable/

---

**END OF DOCUMENT**

*Document ini merupakan kontrak kerja yang mengikat. Perubahan apapun memerlukan persetujuan kedua belah pihak.*
