import streamlit as st
import pandas as pd
import numpy as np

st.header('Stroke Risk Analysis and Prediction')
st.write('**Pelatihan Data Science 1.0** - Universitas Muhammadiyah Semarang')
st.write('Semarang, 13 Desember 2025')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['About Dataset', 
                            'Dashoboards', 
                            'Machine Learning',
                            'Prediction App',
                            'Contact Me'])

with tab1:
    import about
    about.about_dataset()

with tab2:
    import visualisasi
    visualisasi.chart()

with tab3:
    import machine_learning
    machine_learning.ml_model()

with tab4:
    import prediction
    prediction.prediction_app()

