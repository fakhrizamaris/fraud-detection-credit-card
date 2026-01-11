"""
Machine Learning Tab - Model training process explanation
"""
import streamlit as st
import pandas as pd
import altair as alt


def render(model, feature_columns, load_data_func):
    """
    Render tab Machine Learning
    
    Args:
        model: Trained model
        feature_columns: List of feature column names
        load_data_func: Function to load dataset
    """
    st.title("Machine Learning Pipeline")
    st.markdown("### Fraud Detection Model Training Process")
    st.markdown("---")
    
    # 1. Data Preprocessing
    st.markdown("## 1️⃣ Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Preprocessing Steps
        
        **a. Handling Missing Values**
        - Identify and fill missing values
        - Strategy: Mode for categorical, Median for numerical
        
        **b. Feature Engineering**
        - Extract `hour` from transaction timestamp
        - Calculate `age` from date of birth
        - Create `amt_per_hour_ratio` for anomaly detection
        - Create `is_weekend` from transaction day
        
        **c. Categorical Encoding**
        - Label Encoding for: `category`, `gender`, `state`
        - Preserve mapping for new predictions
        """)
    
    with col2:
        st.markdown("""
        ### 🔢 Data Normalization
        
        Numerical features are normalized using **StandardScaler**:
        
        ```
        z = (x - μ) / σ
        ```
        
        Where:
        - `x` = original value
        - `μ` = mean
        - `σ` = standard deviation
        
        **Normalized features:**
        - `amt` (transaction amount)
        - `age` (cardholder age)
        - `hour` (transaction hour)
        - `amt_per_hour_ratio`
        """)
    
    st.markdown("---")
    
    # 2. Class Imbalance
    st.markdown("## 2️⃣ Handling Class Imbalance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### ⚠️ Imbalance Problem
        
        Fraud detection datasets are typically **imbalanced**:
        - **Normal** transactions: ~99%
        - **Fraud** transactions: ~1%
        
        Without handling, the model will:
        - Always predict "Normal"
        - Fail to detect fraud
        - Have low Recall
        """)
        
        # Class Distribution Visualization
        try:
            df_temp = load_data_func()
            class_counts = df_temp['is_fraud'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            class_counts['Class'] = class_counts['Class'].map({0: 'Normal', 1: 'Fraud'})
            
            pie_class = alt.Chart(class_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Class", type="nominal", scale=alt.Scale(
                    domain=['Normal', 'Fraud'],
                    range=['#4CAF50', '#F44336']
                )),
                tooltip=['Class', 'Count']
            ).properties(height=250, title='Class Distribution (Before SMOTE)')
            st.altair_chart(pie_class, use_container_width=True)
        except:
            st.info("Dataset not available for visualization")
    
    with col2:
        st.markdown("""
        ### ✅ Solution: SMOTE
        
        **Synthetic Minority Over-sampling Technique (SMOTE)**:
        
        1. Select minority samples (Fraud)
        2. Find k-nearest neighbors
        3. Create new synthetic samples
        4. Balance class distribution
        
        **Result:**
        - Normal class: 50%
        - Fraud class: 50%
        
        ⚡ Model can learn fraud patterns better!
        """)
        
        # After SMOTE Visualization
        balanced_data = pd.DataFrame({
            'Class': ['Normal', 'Fraud'],
            'Count': [5000, 5000]  # Simulated balanced
        })
        
        pie_balanced = alt.Chart(balanced_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Class", type="nominal", scale=alt.Scale(
                domain=['Normal', 'Fraud'],
                range=['#4CAF50', '#F44336']
            )),
            tooltip=['Class', 'Count']
        ).properties(height=250, title='Class Distribution (After SMOTE)')
        st.altair_chart(pie_balanced, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Model Training
    st.markdown("## 3️⃣ Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌲 Random Forest Classifier
        
        **Why Random Forest?**
        - Robust against overfitting
        - Can handle non-linear data
        - Provides feature importance
        - High performance for classification
        
        **Hyperparameters:**
        """)
        
        st.code("""
model = RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=15,            # Maximum depth of trees
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    random_state=42,         # Reproducibility
    n_jobs=-1,               # Use all CPU cores
    verbose=0                # Disable verbose output
)
        """, language="python")
    
    with col2:
        st.markdown("""
        ### 📈 Training Process
        
        **1. Train-Test Split**
        - Training: 80%
        - Testing: 20%
        - Stratified sampling
        
        **2. Cross-Validation**
        - 5-Fold CV
        - Evaluate performance consistency
        
        **3. Evaluation Metrics**
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - ROC-AUC
        """)
    
    st.markdown("---")
    
    # 4. Feature Importance Explanation
    st.markdown("## 4️⃣ Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        feature_imp_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Altair bar chart
        importance_chart = alt.Chart(feature_imp_df).mark_bar().encode(
            x=alt.X('Importance:Q', title='Importance Score'),
            y=alt.Y('Feature:N', sort='-x', title='Feature'),
            color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
        ).properties(height=300, title='Feature Importance - Random Forest')
        st.altair_chart(importance_chart, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **amt (Amount)**: Transaction amount is the strongest predictor
        - **hour**: Transaction hour affects fraud probability
        - **age**: Cardholder age is also significant
        - **category**: Merchant type influences fraud patterns
        """)
    
    st.markdown("---")
    st.success("✅ Model has been trained and is ready for predictions!")
