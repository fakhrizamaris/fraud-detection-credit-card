# 📊 Fraud Detection Model Presentation Notes

## 1. Model Overview

- **Algorithm**: Random Forest Classifier
- **Objective**: Detect fraudulent transactions in credit card data.
- **Key Strategy**: Used a pre-balanced dataset (Under-sampling/Over-sampling applied before training) to ensure the model learns both classes equally.

## 2. Performance Metrics

The model achieved exceptional performance on the test set:

- **Accuracy**: ~97.2% (Overall correctness)
- **Recall**: ~97.4% (Ability to catch actual fraud) -> _Most critical metric for fraud detetion_
- **Precision**: ~97.0% (Trustworthiness of fraud alerts)
- **ROC-AUC**: ~0.99 (Excellent separation between Fraud and Safe transactions)

_> **Note on High Accuracy**: The high accuracy (>97%) is partly due to the use of a balanced dataset (50% Fraud, 50% Safe) for training and testing. In a real-world production environment where fraud is rare (<1%), accuracy would naturally be high even for a dummy model, so we would focus more on **Recall** (catching fraud) and **Precision** (minimizing false alarms)._

## 3. Key Predictive Features

The model identified the following as the strongest indicators of fraud:

1.  **Transaction Amount (`amt`)**: Fraudulent transactions often deviate from typical spending amounts.
2.  **Amount per Hour Ratio**: Sudden large spikes in spending relative to the time of day.
3.  **Time of Day (`hour`)**: Fraud often occurs at unusual hours (e.g., late night/early morning).
4.  **Category**: Specific spending categories (e.g., online shopping, grocery vs luxury) have different fraud risks.

## 4. Pipeline Steps

1.  **Data Preprocessing**:
    - Handled missing values.
    - Encoded categorical variables (Gender, Category, State).
    - Scaled numerical features (Amount, Age, etc.).
2.  **Feature Engineering**:
    - Extracted `Hour` from transaction time.
    - Calculated `Age` from Date of Birth.
    - Created interaction features like `amt_per_hour_ratio`.
3.  **Training**:
    - Algorithm: Random Forest (n_estimators=200, max_depth=15).
    - Split: 80% Training, 20% Testing.
4.  **Validation**:
    - **Cross-Validation**: Implemented 5-fold Stratified Cross-Validation to ensure model stability across different data subsets.
    - **Confusion Matrix**: Analyzed False Negatives (Missed Fraud) vs False Positives (False Alarms).

## 5. Deployment

- The trained model is saved as `models/fraud_detection_model.pkl`.
- It includes the Model, Scaler, and Encoders for seamless inference.
- Ready for integration with the **Streamlit Dashboard** (`app.py`).
