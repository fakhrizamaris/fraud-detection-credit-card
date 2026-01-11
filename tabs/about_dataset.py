"""
About Dataset Tab - Information about dataset and fraud detection system
"""
import streamlit as st


def render():
    """Render tab About Dataset"""
    st.title("About Dataset & Fraud Detection")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Background
        
        **Credit card fraud** is one of the most damaging financial crimes 
        in the digital era. Key challenges include:
        
        - 💰 Significant financial losses for both consumers and institutions
        - 📈 Increasing sophistication of fraud techniques
        - ⏱️ Traditional manual detection methods are slow and inefficient
        
        **Machine Learning-based** fraud detection systems can identify 
        suspicious transactions in **milliseconds**, significantly 
        reducing financial losses and improving security.
        """)
        
        st.markdown("""
        ### 📊 About the Dataset
        
        This dataset contains **credit card transactions** with the following features:
        
        | Feature | Description |
        |---------|-------------|
        | `category` | Merchant type (grocery, gas, shopping, etc.) |
        | `amt` | Transaction amount in USD |
        | `gender` | Cardholder's gender |
        | `state` | US state where transaction occurred |
        | `age` | Cardholder's age |
        | `hour` | Hour when transaction was made |
        | `is_weekend` | Whether transaction was on weekend |
        | `is_fraud` | Label (0 = Normal, 1 = Fraud) |
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Methodology
        
        This system uses **Random Forest Classifier** algorithm with the following process:
        
        1. **Data Preprocessing**
           - Categorical variable encoding
           - Numerical feature normalization
           - Feature engineering (amt_per_hour_ratio)
        
        2. **Model Training**
           - Data split: 80% training, 20% testing
           - Hyperparameter tuning
           - Cross-validation
        
        3. **Evaluation**
           - Accuracy, Precision, Recall, F1-Score
           - ROC-AUC to measure model discrimination
        """)
        
        st.markdown("""
        ### 💡 System Benefits
        
        - ✅ **Real-time Detection**: Analyze transactions in seconds
        - ✅ **High Accuracy**: Model trained with thousands of historical data
        - ✅ **Loss Prevention**: Identify fraud before it occurs
        - ✅ **Operational Efficiency**: Reduce manual review workload
        """)
    
    st.markdown("---")
    st.info("💡 **Tips**: Use the **Fraud Detection** tab to analyze new transactions, or check **Data Insights** for historical data exploration.")
