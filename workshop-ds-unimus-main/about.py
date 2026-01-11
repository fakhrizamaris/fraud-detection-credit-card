import streamlit as st
import pandas as pd

def about_dataset():
    st.write('**Tentang Dataset**')
    col1, col2= st.columns([5,5])

    with col1:
        link = "https://myhealthcentre.ca/wp-content/uploads/2025/03/17096-819x583.jpg"
        st.image(link, caption="Healthcare Dataset")

    with col2:
        st.write('Menurut Organisasi Kesehatan Dunia (WHO), ' \
        'stroke merupakan penyebab kematian terbanyak kedua di dunia, ' \
        'yang bertanggung jawab atas sekitar 11% dari total kematian. ' \
        'Dataset ini digunakan untuk memprediksi kemungkinan pasien terkena stroke ' \
        'berdasarkan parameter input seperti jenis kelamin, usia, ' \
        'berbagai penyakit, dan status merokok. Setiap baris dalam data ' \
        'memberikan informasi yang relevan tentang pasien.')