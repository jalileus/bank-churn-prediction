# Bank Customer Churn Prediction

## Overview
Machine learning model to predict which bank customers are likely to churn (leave the bank) and enable proactive retention efforts.

## Dataset
- **Source**: [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Size**: 10,000 customers, 12 features
- **Target**: Churn (binary)
- **Churn Rate**: 20.4%

## Features
- `customer_id`: Customer ID
- `credit_score`: Customer's credit score
- `country`: Customer's country (France, Spain, Germany)
- `gender`: Customer's gender
- `age`: Customer's age
- `tenure`: Years with the bank
- `balance`: Account balance
- `products_number`: Number of bank products
- `credit_card`: Has credit card (0/1)
- `active_member`: Active member status (0/1)
- `estimated_salary`: Estimated annual salary
- `churn`

## Methodology

### Data Preprocessing
- Dropping 'customer_id'
- One-hot encoding for categorical features (country, gender)
- Standard scaling for numerical features
- SMOTE for handling class imbalance
- 80/20 train-test split with stratification

### Models Evaluated
| Model                 | AUC-ROC   | Accuracy  | F1-Score  |
| --------------------- | --------- | --------- | --------- |
| Logistic Regression   | 0.766     | 71.5%     | 0.487     |
| Decision Tree         | 0.782     | 78.4%     | 0.553     |
| Random Forest         | 0.855     | 82.3%     | 0.605     |
| **Gradient Boosting** | **0.865** | **83.2%** | **0.620** |
| XGBoost               | 0.860     | 85.4%     | 0.636     |

### Best Model: Gradient Boosting Classifier
- **AUC-ROC**: 0.8654
- **Cross-validation AUC**: 0.9354 (±0.0040)
- **Recall (Churn)**: 67.3%
- **Precision (Churn)**: 57.4%

## Key Findings

### Top Features Driving Churn:
1. Age (older customers more likely to churn)
2. Balance (higher balance → lower churn risk)
3. Number of Products (≥3 products → higher churn)
4. Active Member status (inactive → higher churn)
5. Credit Score (lower scores → higher churn)

### Business Impact (Annual Projection for 10,000 customers):
- **Customers saved**: 1,372
- **Revenue saved**: $686,000
- **Retention campaign cost**: $33,350
- **Net Savings**: $652,650
- **ROI**: Every $1 spent saves $20.57

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- kagglehub
- joblib

## Repository Structure
├── bank_churn.ipynb          # Main analysis notebook
├── churn_model.pkl           # Trained Gradient Boosting model
├── scaler.pkl                # Fitted StandardScaler
├── feature_names.pkl         # Feature names for inference
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore rules

## Author
Alhajali Jalal

## License
This project is for educational/business purposes.


