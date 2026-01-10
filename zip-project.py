# Compress Project ke ZIP
# Dapat digunakan di Python script atau Jupyter Notebook
import zipfile
import os

def create_project_zip(source_dir, output_filename):
    """
    Compress seluruh project menjadi ZIP
    """
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Skip folder yang tidak perlu
            dirs[:] = [d for d in dirs if d not in ['.venv', '__pycache__', '.git', '.ipynb_checkpoints']]

            for file in files:
                # Skip file .zip agar tidak circular/corrupt
                if file.endswith('.zip'):
                    print(f"⏭️  Skipped: {file} (ZIP file)")
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
                print(f"✅ Added: {arcname}")

    print(f"\n📦 ZIP file created: {output_filename}")
    print(f"📊 Size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

# Jalankan
create_project_zip('.', 'fraud-detection-project.zip')