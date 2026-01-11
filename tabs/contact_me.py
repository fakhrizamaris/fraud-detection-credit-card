"""
Contact Me Tab - Profil pengembang dan informasi kontak
"""
import streamlit as st


def render():
    """Render tab Contact Me"""
    st.title("Contact Information")
    st.markdown("---")
    
    # Author Section
    st.markdown("## Penulis")
    
    st.markdown("""
    **[Nama Lengkap Anda]**  
    *Mahasiswa / Data Science Enthusiast*
    
    Jurusan [Nama Jurusan]  
    [Nama Universitas]  
    [Kota, Indonesia]
    """)
    
    st.markdown("---")
    
    # Contact Details
    st.markdown("## Detail Kontak")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Alamat Email**  
        email@university.ac.id
        
        **Telepon**  
        +62 xxx-xxxx-xxxx
        
        **Lokasi**  
        [Kota, Provinsi, Indonesia]
        """)
    
    with col2:
        st.markdown("""
        **LinkedIn**  
        [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
        
        **GitHub**  
        [github.com/yourusername](https://github.com/yourusername)
        
        **Instagram**  
        [@username](https://instagram.com/username)
        """)
    
    st.markdown("---")
    
    # Academic Background
    st.markdown("## Latar Belakang Akademik")
    
    st.markdown("""
    | Jenjang | Institusi | Tahun |
    |---------|-----------|-------|
    | S1 [Nama Program Studi] | [Nama Universitas] | 20XX - Sekarang |
    | SMA | [Nama Sekolah] | 20XX |
    """)
    
    st.markdown("---")
    
    # Research Interests
    st.markdown("## Minat Penelitian")
    
    st.markdown("""
    - Machine Learning & Artificial Intelligence
    - Sistem Deteksi Fraud
    - Data Mining & Pattern Recognition
    - Financial Technology (FinTech)
    """)
    
    st.markdown("---")
    
    # About This Project
    st.markdown("## Tentang Proyek Ini")
    
    st.markdown("""
    **Credit Card Fraud Detection System** ini dikembangkan sebagai bagian dari 
    [nama mata kuliah / skripsi / proyek penelitian] untuk mendemonstrasikan 
    penerapan teknik machine learning dalam keamanan finansial.
    
    **Tujuan:**
    1. Mengimplementasikan classifier Random Forest untuk deteksi fraud
    2. Menangani dataset tidak seimbang menggunakan teknik SMOTE
    3. Mengembangkan dashboard web interaktif menggunakan Streamlit
    4. Menyediakan kemampuan analisis transaksi real-time
    
    **Teknologi & Tools:**
    - **Bahasa Pemrograman:** Python 3.x
    - **Library ML:** Scikit-learn, Imbalanced-learn
    - **Web Framework:** Streamlit
    - **Visualisasi:** Altair, Matplotlib
    - **Pemrosesan Data:** Pandas, NumPy
    """)
    
    st.markdown("---")
    
    # Acknowledgments
    st.markdown("## Ucapan Terima Kasih")
    
    st.markdown("""
    Saya ingin mengucapkan terima kasih kepada:
    - [Nama Dosen Pembimbing] atas bimbingan dan arahan
    - [Nama Universitas/Institusi] atas penyediaan sumber daya
    - Komunitas open-source atas tools dan library yang luar biasa
    """)