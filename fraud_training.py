import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. LOAD DATA
# ========================================
print("📂 Loading dataset...")
df = pd.read_csv('credit_card_transactions2.csv')
print(f"✅ Total data: {len(df)} rows\n")

# ========================================
# 2. FEATURE ENGINEERING
# ========================================
print("🔧 Feature Engineering...")

# Hitung umur dari DOB
df['dob'] = pd.to_datetime(df['dob'])
current_year = datetime.now().year
df['age'] = current_year - df['dob'].dt.year

# Extract jam dari trans_date_trans_time
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour

# Feature tambahan: deteksi pola weekend (fraud sering terjadi di akhir pekan)
df['is_weekend'] = df['trans_date_trans_time'].dt.dayofweek.isin([5, 6]).astype(int)

# Feature tambahan: amount per hour (pola transaksi besar di jam aneh)
df['amt_per_hour_ratio'] = df['amt'] / (df['hour'] + 1)  # +1 untuk hindari divide by zero

# Drop kolom yang tidak relevan
drop_cols = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'trans_num', 
             'unix_time', 'trans_date_trans_time', 'dob', 'merchant', 'job', 
             'zip', 'lat', 'long', 'merch_lat', 'merch_long', 'merch_zipcode', 'city_pop']

# Opsional: drop 'city' dan 'state' jika terlalu banyak unique values (cardinality tinggi)
# Untuk kesederhanaan, kita drop city karena 1000+ unique values
drop_cols.append('city')

df = df.drop(columns=drop_cols, errors='ignore')
print(f"✅ Features setelah engineering: {df.columns.tolist()}\n")

# ========================================
# 3. PREPROCESSING
# ========================================
print("⚙️ Preprocessing...")

# Pisahkan fitur dan target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Label Encoding untuk semua kolom kategorikal
categorical_cols = ['category', 'gender', 'state']
label_encoders = {}

for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded: {col} ({len(le.classes_)} unique values)")

# Scaling untuk numerical features (penting untuk tree-based models)
numerical_cols = ['amt', 'age', 'hour', 'is_weekend', 'amt_per_hour_ratio']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print(f"✅ Total features: {X.shape[1]}\n")

# ========================================
# 4. SPLIT DATA
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}\n")

# ========================================
# 5. TRAINING MODEL
# ========================================
print("🚀 Training XGBoost Classifier...")

# XGBoost lebih baik untuk fraud detection (handling pattern lebih kompleks)
model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)
print("✅ Training selesai!\n")

# ========================================
# 6. EVALUATION
# ========================================
print("📈 Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("="*50)
print("🎯 HASIL EVALUASI MODEL")
print("="*50)
print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f} (dari prediksi fraud, {precision*100:.2f}% benar)")
print(f"Recall    : {recall:.4f} (dari fraud asli, {recall*100:.2f}% terdeteksi)")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("="*50)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔍 Confusion Matrix:")
print(f"   True Negative  : {cm[0][0]} (Benar prediksi Aman)")
print(f"   False Positive : {cm[0][1]} (Salah prediksi Fraud padahal Aman)")
print(f"   False Negative : {cm[1][0]} (Salah prediksi Aman padahal Fraud) ⚠️")
print(f"   True Positive  : {cm[1][1]} (Benar prediksi Fraud)")
print()

# ========================================
# 7. SAVE MODEL & PREPROCESSORS
# ========================================
print("💾 Saving model dan preprocessors...")

# Save semua komponen yang dibutuhkan untuk prediksi
model_artifacts = {
    'model': model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_columns': X.columns.tolist(),
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols
}

with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("✅ Model berhasil disimpan ke 'fraud_detection_model.pkl'")
print("\n🎉 TRAINING SELESAI! Siap digunakan di Streamlit app.")
