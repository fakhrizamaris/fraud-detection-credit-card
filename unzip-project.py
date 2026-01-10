# Extract ZIP Project
# Dapat digunakan di Python script atau Jupyter Notebook
import zipfile
import os

# Nama file ZIP yang akan di-extract
zip_filename = 'fraud-detection-project.zip'

# Cek apakah file ZIP ada
if not os.path.exists(zip_filename):
    print(f"❌ Error: File '{zip_filename}' tidak ditemukan!")
    print(f"📁 Files di folder ini: {os.listdir('.')}")
else:
    # Extract langsung ke folder current (tanpa rename)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    print(f"✅ Project berhasil di-extract dari '{zip_filename}'!")
    print(f"📂 Lokasi: {os.path.abspath('.')}")
    
    # Hitung total files yang di-extract
    total_files = sum([len(files) for root, dirs, files in os.walk('.')])
    print(f"📁 Total files di folder ini: {total_files} files")