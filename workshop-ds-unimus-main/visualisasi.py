import streamlit as st
import pandas as pd
import altair as alt

def chart():
    df = pd.read_excel('healthcare-dataset-stroke-data.xlsx')
    pasien_count = df.shape[0]
    pasien_stroke = df['stroke'].sum()
    stroke_rate = (pasien_stroke / pasien_count) * 100

    #Card Metrics dan Button Filter
    col1, col2, col3, col4, col5, col6 = st.columns([2,2,3,2,2,1])
    with col1:
        st.metric(label="Total Pasien", value = pasien_count)
    with col2:
        st.metric(label="Pasien Stroke", value = pasien_stroke)
    with col3:
        st.metric(label="Persentase", value = f"{stroke_rate:.2f}%")
    
    # Initialize session state untuk filter
    if 'selected_gender' not in st.session_state:
        st.session_state.selected_gender = None
    if 'selected_stroke' not in st.session_state:
        st.session_state.selected_stroke = None
    
    with col4:
        st.write('**Jenis Kelamin**')
        if st.button("Laki-laki"):
            st.session_state.selected_gender = 'Male'
        if st.button("Perempuan"):
            st.session_state.selected_gender = 'Female'
    with col5:
        st.write('**Status**')
        if st.button("Stroke"):
            st.session_state.selected_stroke = 1
        if st.button("No Stroke"):
            st.session_state.selected_stroke = 0
    
    # Reset button
    with col6:
        if st.button("🔄"):
            st.session_state.selected_gender = None
            st.session_state.selected_stroke = None
            st.rerun()
    
    # Apply filter
    filtered_df = df.copy()
    if st.session_state.selected_gender:
        filtered_df = filtered_df[filtered_df['gender'] == st.session_state.selected_gender]
    if st.session_state.selected_stroke is not None:
        filtered_df = filtered_df[filtered_df['stroke'] == st.session_state.selected_stroke]
    
    # Update metrics dengan filtered data
    filtered_count = filtered_df.shape[0]
    filtered_stroke = filtered_df['stroke'].sum()
    filtered_rate = (filtered_stroke / filtered_count * 100) if filtered_count > 0 else 0

    st.dataframe(df.head(5))
    
    #Pie Chart
    col1, col2 = st.columns([5,5])
    def pie_chart_from_counts(counts_df, label_col, value_col, title):
        # counts_df must have label_col and value_col
        chart = alt.Chart(counts_df).mark_arc(innerRadius=40).encode(
            theta=alt.Theta(field=value_col, type='quantitative'),
            color=alt.Color(field=label_col, type='nominal', sort=alt.EncodingSortField(field=value_col, order='descending')),
            tooltip=[alt.Tooltip(label_col+':N'), alt.Tooltip(value_col+':Q')]
        ).properties(height=300, title=title)
        return chart
    with col1:
    #Persentase Jenis Kelamin
        gender_counts = filtered_df['gender'].fillna('Unknown').value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        chart_gender = pie_chart_from_counts(gender_counts, 'gender', 'count', 'Jenis Kelamin')
        st.altair_chart(chart_gender, use_container_width=True)
    
    with col2:
    #Persentase Hipertensi
        hyper_counts = filtered_df['hypertension'].fillna(0).map({0: 'Tidak', 1: 'Hipertensi'}).value_counts().reset_index()
        hyper_counts.columns = ['hypertension', 'count']
        chart_hyper = pie_chart_from_counts(hyper_counts, 'hypertension', 'count', 'Hipertensi pada Pasien')
        st.altair_chart(chart_hyper, use_container_width=True)
    
    col1, col2 = st.columns([5,5])
    with col1:
    #Persentase Riwayat Merokok
        smoke_counts = filtered_df['smoking_status'].fillna('Unknown').value_counts().reset_index()
        smoke_counts.columns = ['smoking_status', 'count']
        chart_smoke = pie_chart_from_counts(smoke_counts, 'smoking_status', 'count', 'Riwayat Merokok pada Pasien')
        st.altair_chart(chart_smoke, use_container_width=True)
    
    with col2:
    #Persentase Tempat Tinggal
        residence_counts = filtered_df['Residence_type'].fillna('Unknown').value_counts().reset_index()
        residence_counts.columns = ['Residence_type', 'count']
        chart_res = pie_chart_from_counts(residence_counts, 'Residence_type', 'count', 'Tipe Tempat Tinggal Pasien')
        st.altair_chart(chart_res, use_container_width=True)    

    #Histogram BMI
    st.write("**Distribusi BMI Pasien**")
    bmi_chart = alt.Chart(filtered_df).mark_bar().encode(
            alt.X('bmi:Q', bin=alt.Bin(maxbins=30), title='BMI'),
            alt.Y('count():Q', title='Frekuensi'),
            tooltip=[alt.Tooltip('count():Q', title='Frekuensi')]
        ).properties(height=300)
    st.altair_chart(bmi_chart, use_container_width=True)

    #Scatter Plot
    #Glukosa vs BMI
    st.write("**Hubungan BMI & Rata-rata Glukosa**")
    scatter1 = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X('avg_glucose_level:Q', title='Rata-rata Glukosa'),
            y=alt.Y('bmi:Q', title='BMI'),
            color=alt.Color('stroke:N', title='Stroke', scale=alt.Scale(scheme='viridis')),
            tooltip=['age', 'bmi', 'avg_glucose_level', 'stroke']
        ).interactive().properties(height=320)
    st.altair_chart(scatter1, use_container_width=True)
    
    #Usia vs BMI
    st.write("**Hubungan BMI & Usia**")
    scatter2 = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X('age:Q', title='Usia'),
            y=alt.Y('bmi:Q', title='BMI'),
            color=alt.Color('stroke:N', title='Stroke', scale=alt.Scale(scheme='viridis')),
            tooltip=['age', 'bmi', 'avg_glucose_level', 'stroke']
        ).interactive().properties(height=320)
    st.altair_chart(scatter2, use_container_width=True)


    #Distribusi Stroke dengan Box Plot
    st.write("**Distribusi Usia Pasien berdasarkan Stroke (Box Plot)**")
    box = alt.Chart(filtered_df).mark_boxplot(extent=1.5).encode(
            x=alt.X('stroke:N', title='Stroke'),
            y=alt.Y('age:Q', title='Usia'),
            color=alt.Color('stroke:N', legend=None),
            tooltip=[alt.Tooltip('count():Q', title='Jumlah')]
        ).properties(height=320)
    st.altair_chart(box, use_container_width=True)

    #Distribusi Usia berdasarkan Kasus Stroke dengan Line Chart
    st.write("**Distribusi Usia berdasarkan Kasus Stroke**")
    age_stroke_counts = filtered_df.groupby(['age', 'stroke']).size().reset_index(name='counts')
    age_stroke_counts['stroke'] = age_stroke_counts['stroke'].astype(str)
    line_age = alt.Chart(age_stroke_counts).mark_line(point=True).encode(
        x=alt.X('age:Q', title='Usia'),
        y=alt.Y('counts:Q', title='Jumlah Pasien'),
        color=alt.Color('stroke:N', title='Stroke'),
        tooltip=['age', 'counts', 'stroke']
    ).properties(height=350)
    st.altair_chart(line_age, use_container_width=True)

    #Perkembangan Penderita Stroke dan Perbandingan Berdasarkan Kategori
    col1, col2 = st.columns([5,5])
    with col1:
    #Berdasarkan BMI Category
        st.write('**Perkembangan Penderita Stroke**')
        bmi_stroke_counts = filtered_df.groupby(['bmi_category', 'stroke']).size().reset_index(name='counts')
        bmi_stroke_counts['stroke_label'] = bmi_stroke_counts['stroke'].map({0: 'No Stroke', 1: 'Stroke'}).astype(str)
        chart_bmi_cat = alt.Chart(bmi_stroke_counts).mark_line(point=True).encode(
            x=alt.X('bmi_category:N', sort=alt.EncodingSortField(field='counts', op='sum', order='descending'), title='BMI Category'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Stroke'),
            tooltip=['bmi_category', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_bmi_cat, use_container_width=True)

    
    #Berdasarkan kolom 'age_category'
        age_cat_stroke_counts = filtered_df.groupby(['age_category', 'stroke']).size().reset_index(name='counts')
        age_cat_stroke_counts['stroke_label'] = age_cat_stroke_counts['stroke'].map({0: 'No Stroke', 1: 'Stroke'}).astype(str)
        chart_age_cat = alt.Chart(age_cat_stroke_counts).mark_line(point=True).encode(
            x=alt.X('age_category:N', title='Age Category'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Stroke'),
            tooltip=['age_category', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_age_cat, use_container_width=True)
    
    #Berdasarkan Resiko Stroke
        risk_stroke_counts = filtered_df.groupby(['risk_score', 'stroke']).size().reset_index(name='counts')
        risk_stroke_counts['stroke_label'] = risk_stroke_counts['stroke'].map({0: 'No Stroke', 1: 'Stroke'}).astype(str)
        chart_risk = alt.Chart(risk_stroke_counts).mark_line(point=True).encode(
            x=alt.X('risk_score:O', title='Risk Score'),  # treat as ordinal if discrete
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Stroke'),
            tooltip=['risk_score', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_risk, use_container_width=True)
    
    with col2:
        st.write('**Perbandingan Berdasarkan Kategori**')
    #Berdasarkan Tipe Pekerjaan
        work_stroke_counts = filtered_df.groupby(['work_type', 'stroke']).size().reset_index(name='counts')
        work_stroke_counts['stroke_label'] = work_stroke_counts['stroke'].map({0: 'No Stroke', 1: 'Stroke'}).astype(str)
        chart_work = alt.Chart(work_stroke_counts).mark_bar().encode(
            x=alt.X('work_type:N', title='Tipe Pekerjaan'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Stroke'),
            tooltip=['work_type', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_work, use_container_width=True)
    
    #Berdasarkan Kategori Glukosa
        glucose_stroke_counts = filtered_df.groupby(['glucose_category', 'stroke']).size().reset_index(name='counts')
        glucose_stroke_counts['stroke_label'] = glucose_stroke_counts['stroke'].map({0: 'No Stroke', 1: 'Stroke'}).astype(str)
        chart_glucose = alt.Chart(glucose_stroke_counts).mark_bar().encode(
            x=alt.X('glucose_category:N', title='Kategori Glukosa'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Stroke'),
            tooltip=['glucose_category', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_glucose, use_container_width=True)
    
    #Berdasarkan Life Style Risk
        lifestyle_stroke_counts = filtered_df.groupby(['lifestyle_risk', 'stroke']).size().reset_index(name='counts')
        lifestyle_stroke_counts['stroke_label'] = lifestyle_stroke_counts['stroke'].map({0: 'No Stroke', 1: 'Stroke'}).astype(str)
        chart_lifestyle = alt.Chart(lifestyle_stroke_counts).mark_bar().encode(
            x=alt.X('lifestyle_risk:N', title='Life Style Risk'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Stroke'),
            tooltip=['lifestyle_risk', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_lifestyle, use_container_width=True)