import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#1. Membaca Dataset
st.write('**1. Dataset**')
st.dataframe(df.head())

#2. Visualisasi Data - Jumlah Pasien Berdasarkan Usia
st.write('**2. Visualisasi Data - Jumlah Pasien Berdasarkan Usia**')
st.line_chart(df['age'].value_counts().sort_index())

#3. Visualisasi Data - Tipe Pekerjaan
st.write('**3. Visualisasi Data - Tipe Pekerjaan**')
st.bar_chart(df['work_type'].value_counts())

#4. Visualisasi Data - Jenis Kelamin
st.write('**4. Visualisasi Data - Jenis Kelamin**')
category_df = df['gender'].value_counts(dropna=False).reset_index()
category_df.columns = ['gender', 'count']
fig = px.pie(category_df, names='gender', values='count')
st.plotly_chart(fig, use_container_width=True)

#5. Interaktif Komponen
st.write('**5. Button**')
st.button("Reset", type="primary")
if st.button("Say hello"):
    st.write("Why hello there")
else:
    st.write("Goodbye")

if st.button("Aloha", type="tertiary"):
    st.write("Ciao")

#6. Checkbox
st.write('**6. Checkbox**')
agree = st.checkbox("I agree")
if agree:
    st.write("Great!")

#7. Multiselect
st.write('**7. Multiselect**')
options = st.multiselect(
    "What are your favorite colors",
    ["Green", "Yellow", "Red", "Blue"],
    ["Yellow", "Red"],
)
st.write("You selected:", options)

#8. Slider
st.write('**8. Slider**')
start_tyres, end_tyres = st.select_slider(
    "Pilih komponen ban untuk race pekan ini",
    options=[
        "Hyper Soft",
        "Ultrs Soft",
        "Super Soft",
        "Soft",
        "Medium",
        "Hard",
        "Super Hard",
    ],
    value=("Hyper Soft", "Soft"),
)
st.write("Anda memilih", start_tyres, "dan", end_tyres)

#9. Toggle
st.write('**9. Toggle**')
on = st.toggle("Activate feature")
if on:
    st.write("Feature activated!")

#10. Number Input
st.write('**10. Number Input**')
number = st.number_input(
    "Insert a number", value=None, placeholder="Type a number..."
)
st.write("The current number is ", number)

#11. Date Input
st.write('**11. Date Input**')
import datetime
d = st.date_input("When's your birthday", datetime.date(1999, 9, 14))
st.write("Your birthday is:", d)

#12 Image
st.write('**12. Input Image**')
link = "https://unair.ac.id/wp-content/uploads/2023/04/Foto-by-Kompas-Money.jpg"
st.image(link, caption="Rumah Sakit")

#13. Membuat Kolom
st.write('**13. Membuat Kolom**')
col1, col2= st.columns(2)

with col1:
    st.write('**Jumlah Pasien Berdasarkan Usia**')
    st.line_chart(df['age'].value_counts().sort_index())

with col2:
    st.write('**Visualisasi Data - Tipe Pekerjaan**')
    st.bar_chart(df['work_type'].value_counts())

#14. Membuat tab
st.write('**14. Membuat Tab**')
tab1, tab2= st.tabs(["Line", "Bar"])

with tab1:
    st.write('**Jumlah Pasien Berdasarkan Usia**')
    st.line_chart(df['age'].value_counts().sort_index())
with tab2:
    st.write('**Visualisasi Data - Tipe Pekerjaan**')
    st.bar_chart(df['work_type'].value_counts())

#15. Expander
st.write('**15. Expander**')
with st.expander("See explanation"):
    st.write('''
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    ''')
    st.image("https://static.streamlit.io/examples/dice.jpg")




