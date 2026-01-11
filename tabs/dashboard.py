"""
Data Insights Tab - Visualisasi eksplorasi data historis dengan EDA lengkap
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


def render(load_data_func):
    """
    Render tab Data Insights dengan EDA lengkap
    
    Args:
        load_data_func: Function to load dataset
    """
    st.title("📈 Data Insights Dashboard")
    st.markdown("### Historical Data Exploration & In-Depth Analysis")
    st.markdown("---")
    
    try:
        df_raw = load_data_func()
        
        # --- FEATURE ENGINEERING untuk visualisasi ---
        df = df_raw.copy()
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['dob'] = pd.to_datetime(df['dob'])
        today = pd.Timestamp.today()
        df['age'] = ((today - df['dob']).dt.days / 365.25).astype(int)
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # ==============================================
        # SECTION 1: OVERVIEW METRICS
        # ==============================================
        st.markdown("## 1️⃣ Overview Dataset")
        
        total_trx = len(df)
        total_fraud = df['is_fraud'].sum()
        fraud_rate = (total_fraud / total_trx) * 100
        avg_amount = df['amt'].mean()
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Total Transaksi", f"{total_trx:,}")
        with m_col2:
            st.metric("Total Fraud", f"{total_fraud:,}")
        with m_col3:
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with m_col4:
            st.metric("Rata-rata Amount", f"${avg_amount:,.2f}")
        with m_col5:
            st.metric("Jumlah Fitur", f"{len(df.columns)}")
        
        # Sample Data
        st.markdown("#### 📋 Sample Data (5 Baris Pertama)")
        display_cols = ['trans_date_trans_time', 'category', 'amt', 'gender', 'state', 'age', 'hour', 'is_fraud']
        st.dataframe(df[display_cols].head(), width='stretch')
        
        st.markdown("---")
        
        # ==============================================
        # SECTION 2: DATA QUALITY & MISSING VALUES
        # ==============================================
        st.markdown("## 4️⃣ Data Quality Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Missing Values Analysis")
            # Check missing values
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, width='stretch')
            else:
                st.success("✅ No missing values found in the dataset!")
        
        with col2:
            st.markdown("#### 📊 Data Types Overview")
            dtype_counts = df.dtypes.value_counts()
            dtype_df = pd.DataFrame({
                'Data Type': dtype_counts.index.astype(str),
                'Count': dtype_counts.values
            })
            
            dtype_chart = alt.Chart(dtype_df).mark_bar().encode(
                x=alt.X('Count:Q', title='Column Count'),
                y=alt.Y('Data Type:N', sort='-x', title='Data Type'),
                color=alt.Color('Data Type:N', legend=None)
            ).properties(height=200)
            st.altair_chart(dtype_chart, width='stretch')
        
        st.markdown("---")
        
        # ==============================================
        # SECTION 3: OUTLIER DETECTION
        # ==============================================
        st.markdown("## 5️⃣ Outlier Detection")
        st.markdown("Using **IQR (Interquartile Range)** method to detect outliers in numerical features.")
        
        # Select numerical columns for outlier detection
        numerical_cols = ['amt', 'age']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Box Plot - Amount (Before Handling)")
            box_amt = alt.Chart(df).mark_boxplot(extent=1.5).encode(
                y=alt.Y('amt:Q', title='Amount (USD)'),
                color=alt.value('#3498db')
            ).properties(height=300, title='Amount Distribution with Outliers')
            st.altair_chart(box_amt, width='stretch')
            
            # Calculate outliers
            Q1_amt = df['amt'].quantile(0.25)
            Q3_amt = df['amt'].quantile(0.75)
            IQR_amt = Q3_amt - Q1_amt
            lower_amt = Q1_amt - 1.5 * IQR_amt
            upper_amt = Q3_amt + 1.5 * IQR_amt
            outliers_amt = df[(df['amt'] < lower_amt) | (df['amt'] > upper_amt)]
            
            st.info(f"""
            **Amount Statistics:**
            - Q1: ${Q1_amt:,.2f}
            - Q3: ${Q3_amt:,.2f}
            - IQR: ${IQR_amt:,.2f}
            - Lower Bound: ${lower_amt:,.2f}
            - Upper Bound: ${upper_amt:,.2f}
            - **Total Outliers: {len(outliers_amt):,} ({len(outliers_amt)/len(df)*100:.2f}%)**
            """)
        
        with col2:
            st.markdown("#### 📈 Box Plot - Age")
            box_age = alt.Chart(df).mark_boxplot(extent=1.5).encode(
                y=alt.Y('age:Q', title='Age (Years)'),
                color=alt.value('#e74c3c')
            ).properties(height=300, title='Age Distribution')
            st.altair_chart(box_age, width='stretch')
            
            # Calculate outliers for age
            Q1_age = df['age'].quantile(0.25)
            Q3_age = df['age'].quantile(0.75)
            IQR_age = Q3_age - Q1_age
            lower_age = Q1_age - 1.5 * IQR_age
            upper_age = Q3_age + 1.5 * IQR_age
            outliers_age = df[(df['age'] < lower_age) | (df['age'] > upper_age)]
            
            st.info(f"""
            **Age Statistics:**
            - Q1: {Q1_age:.0f} years
            - Q3: {Q3_age:.0f} years
            - IQR: {IQR_age:.0f} years
            - Lower Bound: {lower_age:.0f} years
            - Upper Bound: {upper_age:.0f} years
            - **Total Outliers: {len(outliers_age):,} ({len(outliers_age)/len(df)*100:.2f}%)**
            """)
        
        st.markdown("---")
        
        # ==============================================
        # SECTION 4: NORMALISASI DATA
        # ==============================================
        st.markdown("## 6️⃣ Data Normalization")
        st.markdown("Comparison of data distribution **before** and **after** normalization using **StandardScaler**.")
        
        from sklearn.preprocessing import StandardScaler
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Before Normalization")
            
            # Amount Distribution Before
            hist_before = alt.Chart(df).mark_bar(opacity=0.7).encode(
                x=alt.X('amt:Q', bin=alt.Bin(maxbins=30), title='Amount (USD)'),
                y=alt.Y('count()', title='Frequency'),
                tooltip=[alt.Tooltip('count()', title='Count')]
            ).properties(height=200, title='Distribusi Amount (Original)')
            st.altair_chart(hist_before, width='stretch')
            
            # Stats before
            st.markdown(f"""
            **Statistik Amount (Original):**
            - Mean: ${df['amt'].mean():,.2f}
            - Std: ${df['amt'].std():,.2f}
            - Min: ${df['amt'].min():,.2f}
            - Max: ${df['amt'].max():,.2f}
            """)
        
        with col2:
            st.markdown("#### 📊 After Normalization")
            
            # Normalize amount
            scaler = StandardScaler()
            df_normalized = df.copy()
            df_normalized['amt_normalized'] = scaler.fit_transform(df[['amt']])
            
            # Amount Distribution After
            hist_after = alt.Chart(df_normalized).mark_bar(opacity=0.7, color='#2ecc71').encode(
                x=alt.X('amt_normalized:Q', bin=alt.Bin(maxbins=30), title='Amount (Normalized)'),
                y=alt.Y('count()', title='Frequency'),
                tooltip=[alt.Tooltip('count()', title='Count')]
            ).properties(height=200, title='Distribusi Amount (Normalized)')
            st.altair_chart(hist_after, width='stretch')
            
            # Stats after
            st.markdown(f"""
            **Statistik Amount (Normalized):**
            - Mean: {df_normalized['amt_normalized'].mean():.4f}
            - Std: {df_normalized['amt_normalized'].std():.4f}
            - Min: {df_normalized['amt_normalized'].min():.4f}
            - Max: {df_normalized['amt_normalized'].max():.4f}
            """)
        
        st.markdown("---")
        
        # ==============================================
        # SECTION 5: EDA - DISTRIBUSI KATEGORIKAL
        # ==============================================
        st.markdown("## 7️⃣ Exploratory Data Analysis (EDA)")
        
        # Row 1: Gender and Category
        st.markdown("### 📊 Categorical Variable Distribution")
        
        c_col1, c_col2 = st.columns([1, 2])
        
        with c_col1:
            st.markdown("##### 👥 Gender Distribution")
            gender_counts = df['gender'].value_counts().reset_index()
            gender_counts.columns = ['gender', 'count']
            gender_counts['gender'] = gender_counts['gender'].map({'M': 'Male', 'F': 'Female'})
            
            pie_chart = alt.Chart(gender_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="gender", type="nominal", scale=alt.Scale(
                    domain=['Male', 'Female'],
                    range=['#3498db', '#e91e63']
                )),
                tooltip=['gender', 'count']
            ).properties(height=300)
            st.altair_chart(pie_chart, width='stretch')
            
        with c_col2:
            st.markdown("##### 🏢 Top 10 Transaction Categories")
            cat_counts = df['category'].value_counts().head(10).reset_index()
            cat_counts.columns = ['category', 'count']
            cat_counts['category'] = cat_counts['category'].apply(lambda x: x.replace('_', ' ').title())
            
            bar_chart = alt.Chart(cat_counts).mark_bar().encode(
                x=alt.X('count:Q', title='Transaction Count'),
                y=alt.Y('category:N', sort='-x', title='Category'),
                color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['category', 'count']
            ).properties(height=300)
            st.altair_chart(bar_chart, width='stretch')
        
        # Row 2: Hour and Weekend
        st.markdown("### 📊 Transaction Time Patterns")
        
        c_col3, c_col4 = st.columns(2)
        
        with c_col3:
            st.markdown("##### 🕒 Transactions by Hour")
            trx_hour_counts = df['hour'].value_counts().sort_index().reset_index()
            trx_hour_counts.columns = ['hour', 'count']
            
            line_chart = alt.Chart(trx_hour_counts).mark_area(
                interpolate='monotone',
                fillOpacity=0.3,
                line=True
            ).encode(
                x=alt.X('hour:O', title='Hour (0-23)'),
                y=alt.Y('count:Q', title='Transaction Count'),
                tooltip=['hour', 'count']
            ).properties(height=300)
            st.altair_chart(line_chart, width='stretch')
            
        with c_col4:
            st.markdown("##### 📅 Weekday vs Weekend")
            weekend_counts = df['is_weekend'].value_counts().reset_index()
            weekend_counts.columns = ['is_weekend', 'count']
            weekend_counts['label'] = weekend_counts['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
            
            weekend_chart = alt.Chart(weekend_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="label", type="nominal", scale=alt.Scale(
                    domain=['Weekday', 'Weekend'],
                    range=['#4CAF50', '#FF9800']
                )),
                tooltip=['label', 'count']
            ).properties(height=300)
            st.altair_chart(weekend_chart, width='stretch')
        
        st.markdown("---")
        
        # ==============================================
        # SECTION 6: EDA - DISTRIBUSI NUMERIK
        # ==============================================
        st.markdown("### 📊 Numerical Variable Distribution")
        
        c_col5, c_col6 = st.columns(2)
        
        with c_col5:
            st.markdown("##### 💰 Cardholder Age Distribution")
            hist_age = alt.Chart(df).mark_bar(opacity=0.8).encode(
                x=alt.X('age:Q', bin=alt.Bin(maxbins=20), title='Age'),
                y=alt.Y('count()', title='Frequency'),
                color=alt.value('#9b59b6'),
                tooltip=[alt.Tooltip('count()', title='Count')]
            ).properties(height=300)
            st.altair_chart(hist_age, width='stretch')
            
        with c_col6:
            st.markdown("##### 💳 Transaction Amount Distribution")
            # Filter for better visualization (remove extreme outliers)
            df_filtered = df[df['amt'] < df['amt'].quantile(0.99)]
            
            hist_amt = alt.Chart(df_filtered).mark_bar(opacity=0.8).encode(
                x=alt.X('amt:Q', bin=alt.Bin(maxbins=30), title='Amount (USD)'),
                y=alt.Y('count()', title='Frequency'),
                color=alt.value('#1abc9c'),
                tooltip=[alt.Tooltip('count()', title='Count')]
            ).properties(height=300)
            st.altair_chart(hist_amt, width='stretch')
        
        st.markdown("---")
        
        # ==============================================
        # SECTION 7: FRAUD ANALYSIS
        # ==============================================
        st.markdown("## 8️⃣ Fraud Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🚨 Fraud by Category")
            fraud_by_cat = df.groupby('category')['is_fraud'].sum().sort_values(ascending=False).head(10).reset_index()
            fraud_by_cat.columns = ['category', 'fraud_count']
            fraud_by_cat['category'] = fraud_by_cat['category'].apply(lambda x: x.replace('_', ' ').title())
            
            fraud_cat_chart = alt.Chart(fraud_by_cat).mark_bar().encode(
                x=alt.X('fraud_count:Q', title='Fraud Count'),
                y=alt.Y('category:N', sort='-x', title='Category'),
                color=alt.Color('fraud_count:Q', scale=alt.Scale(scheme='reds'), legend=None),
                tooltip=['category', 'fraud_count']
            ).properties(height=300)
            st.altair_chart(fraud_cat_chart, width='stretch')
        
        with col2:
            st.markdown("##### ⏰ Fraud by Hour")
            fraud_by_hour = df.groupby('hour')['is_fraud'].sum().reset_index()
            fraud_by_hour.columns = ['hour', 'fraud_count']
            
            fraud_hour_chart = alt.Chart(fraud_by_hour).mark_line(point=True, color='#e74c3c').encode(
                x=alt.X('hour:O', title='Hour'),
                y=alt.Y('fraud_count:Q', title='Fraud Count'),
                tooltip=['hour', 'fraud_count']
            ).properties(height=300)
            st.altair_chart(fraud_hour_chart, width='stretch')
        
        # Box Plot: Amount by Age Group (Fraud vs Normal)
        st.markdown("##### 📦 Amount Distribution by Age Group (Fraud vs Normal)")
        
        # Create age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 40, 60, 100], 
                                  labels=['Young (18-25)', 'Adult (26-40)', 'Middle (41-60)', 'Senior (60+)'])
        df['fraud_label'] = df['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
        
        # Filter extreme outliers for better visualization
        df_box = df[df['amt'] < df['amt'].quantile(0.95)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Normal transactions box plot
            df_normal = df_box[df_box['fraud_label'] == 'Normal']
            box_normal = alt.Chart(df_normal).mark_boxplot(extent=1.5, color='#3498db').encode(
                x=alt.X('age_group:N', title='Age Group', sort=['Young (18-25)', 'Adult (26-40)', 'Middle (41-60)', 'Senior (60+)']),
                y=alt.Y('amt:Q', title='Amount (USD)'),
                tooltip=['age_group', 'amt']
            ).properties(height=350, title='Normal Transactions')
            st.altair_chart(box_normal, use_container_width=True)
        
        with col2:
            # Fraud transactions box plot
            df_fraud = df_box[df_box['fraud_label'] == 'Fraud']
            box_fraud = alt.Chart(df_fraud).mark_boxplot(extent=1.5, color='#e74c3c').encode(
                x=alt.X('age_group:N', title='Age Group', sort=['Young (18-25)', 'Adult (26-40)', 'Middle (41-60)', 'Senior (60+)']),
                y=alt.Y('amt:Q', title='Amount (USD)'),
                tooltip=['age_group', 'amt']
            ).properties(height=350, title='Fraud Transactions')
            st.altair_chart(box_fraud, use_container_width=True)
        
        st.markdown("""
        **Insight:**
        - Box plots show Amount distribution for each age group
        - **Left (Blue)**: Normal transactions - **Right (Red)**: Fraud transactions
        - Compare median, quartiles, and outliers between Normal vs Fraud
        """)
        
        st.markdown("---")
        
        # ==============================================
        # SECTION 8: CORRELATION HEATMAP
        # ==============================================
        st.markdown("## 9️⃣ Correlation Analysis")
        st.markdown("Correlation heatmap between numerical features using **Pearson Correlation**.")
        
        # Select numerical columns for correlation
        corr_cols = ['amt', 'age', 'hour', 'is_weekend', 'is_fraud']
        corr_df = df[corr_cols].corr()
        
        # Reshape for Altair
        corr_melted = corr_df.reset_index().melt(id_vars='index')
        corr_melted.columns = ['Variable1', 'Variable2', 'Correlation']
        
        # Heatmap
        heatmap = alt.Chart(corr_melted).mark_rect().encode(
            x=alt.X('Variable2:N', title=None),
            y=alt.Y('Variable1:N', title=None),
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                           legend=alt.Legend(title='Correlation')),
            tooltip=[
                alt.Tooltip('Variable1:N', title='Variabel 1'),
                alt.Tooltip('Variable2:N', title='Variabel 2'),
                alt.Tooltip('Correlation:Q', title='Correlation', format='.3f')
            ]
        ).properties(width=400, height=400, title='Correlation Heatmap')
        
        # Text labels - use simple condition based on absolute correlation value
        text = alt.Chart(corr_melted).mark_text(fontSize=14, fontWeight='bold').encode(
            x='Variable2:N',
            y='Variable1:N',
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                (alt.datum.Correlation > 0.5) | (alt.datum.Correlation < -0.5),
                alt.value('white'),
                alt.value('black')
            )
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.altair_chart(heatmap + text, width='stretch')
        
        with col2:
            st.markdown("""
            #### 📊 Correlation Interpretation
            
            **Correlation Scale:**
            - **+1.0**: Perfect positive correlation
            - **0.0**: No correlation
            - **-1.0**: Perfect negative correlation
            
            **Strength Categories:**
            - |r| < 0.3: Weak
            - 0.3 ≤ |r| < 0.7: Moderate
            - |r| ≥ 0.7: Strong
            
            **Data Insights:**
            - `amt` and `is_fraud`: Positive correlation (fraud transactions tend to have higher amounts)
            - `hour` and `is_fraud`: Note specific hour patterns
            - Low correlation between predictors → minimal multicollinearity risk
            """)
        
        st.markdown("---")
        st.success("✅ Data exploration complete! Use these insights to understand fraud patterns in the dataset.")
            
    except FileNotFoundError:
        st.warning("⚠️ Dataset file `data/credit_card_transactions2.csv` not found.")
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
