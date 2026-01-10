# Script untuk update struktur project di README.md
with open('README.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Struktur project yang baru
new_structure = """```
fraud-detection/
├── app.py                          # Streamlit dashboard
├── training_model.py               # Script training model
├── zip-project.py                  # Script compress project ke ZIP
├── unzip-project.py                # Script extract project dari ZIP
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi
├── SCREENSHOT_GUIDE.md             # Panduan penempatan screenshot
├── PRESENTATION_NOTES.md           # Catatan presentasi (jika ada)
├── prd_fraud_detection.md          # PRD (jika ada)
├── data/                           # Dataset
│   └── credit_card_transactions2.csv
├── models/                         # Trained model + preprocessors
│   └── fraud_detection_model.pkl
├── notebook/                       # Jupyter notebook
│   └── Fraud_detection_RF.ipynb
└── screenshots/                    # Screenshot (opsional)
    ├── dashboard/                  # Screenshot dashboard
    ├── notebook/                   # Screenshot notebook  
    ├── results/                    # Screenshot hasil
    └── installation/               # Screenshot instalasi
```
"""

# Find the structure section (starts at line 63, ends at line 75)
# Replace lines 63-75 with new structure
new_lines = lines[:63] + [new_structure + '\n'] + lines[75:]

# Write back
with open('README.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ README.md berhasil diupdate!")
print("📝 Struktur project sudah diperbarui dengan:")
print("   - zip-project.py")
print("   - unzip-project.py")  
print("   - SCREENSHOT_GUIDE.md")
print("   - folder screenshots/")
