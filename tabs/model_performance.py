"""
Model Performance Tab - Model evaluation dashboard
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def render(model, model_info, performance, feature_columns):
    """
    Render tab Model Performance
    
    Args:
        model: Trained model
        model_info: Dict containing model information
        performance: Dict containing performance metrics
        feature_columns: List of feature column names
    """
    st.title("Model Performance Dashboard")
    st.markdown("### Random Forest Model Evaluation")
    st.markdown("---")
    
    # Model Info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤖 Model Information")
        st.markdown(f"**Algorithm:** {model_info.get('algorithm', 'Random Forest')}")
        st.markdown(f"**N Estimators:** {model_info.get('n_estimators', 200)}")
        st.markdown(f"**Max Depth:** {model_info.get('max_depth', 15)}")
        if 'trained_at' in model_info:
            st.markdown(f"**Trained:** {model_info['trained_at']}")
    
    with col2:
        st.markdown("#### 📈 Performance Metrics")
        if performance:
            st.metric("Accuracy", f"{performance.get('accuracy', 0)*100:.2f}%")
            st.metric("Recall", f"{performance.get('recall', 0)*100:.2f}%")
            st.metric("Precision", f"{performance.get('precision', 0)*100:.2f}%")
        else:
            st.info("Performance metrics not available in model file")
    
    with col3:
        st.markdown("#### 🎯 Model Status")
        if performance:
            acc = performance.get('accuracy', 0)
            rec = performance.get('recall', 0)
            
            if acc >= 0.85 and rec >= 0.80:
                st.success("✅ Model Meets Requirements")
            else:
                st.warning("⚠️ Model Below Target")
            
            st.metric("F1-Score", f"{performance.get('f1_score', 0)*100:.2f}%")
            st.metric("ROC-AUC", f"{performance.get('roc_auc', 0):.4f}")
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🔍 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        feature_imp_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show table
        st.markdown("#### 📋 Feature Importance Table")
        st.dataframe(feature_imp_df.style.format({'Importance': '{:.4f}'}), use_container_width=True)
    
    st.markdown("---")
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("### 📜 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Download history
        csv_history = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download All Predictions (CSV)",
            data=csv_history,
            file_name=f'fraud_prediction_history_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    else:
        st.info("📭 No prediction history yet. Make predictions in the 'Fraud Detection' tab first.")
