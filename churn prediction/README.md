# Customer Churn Prediction System

A comprehensive Python-based machine learning system for predicting customer churn in telecommunications companies.

## Overview

This project implements a complete customer churn prediction pipeline using machine learning techniques. It includes data preprocessing, exploratory data analysis, model training, evaluation, and prediction capabilities.

## Features

- **Data Loading & Preprocessing**: Handles missing values, converts data types, and prepares features
- **Exploratory Data Analysis**: Visualizes churn distribution and data patterns
- **Machine Learning Models**: 
  - Logistic Regression (baseline model)
  - Random Forest Classifier (robust ensemble model)
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC AUC
- **Prediction Function**: Ready-to-use function for predicting churn on new customer data

## Dataset

The system is designed to work with the Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). If the dataset is not available, the script will use a small dummy dataset for demonstration purposes.

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd churn-prediction
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Place your dataset file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) in the project directory
2. Run the script:
```bash
python churn_prediction.py
```

## Project Structure

```
churn-prediction/
├── churn_prediction.py    # Main script
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (not included in repo)
```

## Key Components

### Data Preprocessing
- Handles `TotalCharges` column conversion (spaces to NaN, then to float)
- Converts categorical variables using One-Hot Encoding
- Scales numerical features using StandardScaler
- Converts target variable (Churn) from 'Yes'/'No' to 1/0

### Model Training
- **Logistic Regression**: Linear model, good baseline for binary classification
- **Random Forest**: Ensemble method, robust to outliers and non-linear relationships

### Evaluation Metrics
- Accuracy Score
- Precision Score
- Recall Score
- F1-Score
- ROC AUC Score
- Confusion Matrix visualization

### Prediction Function
```python
predict_churn(model, preprocessor, new_customer_data_dict)
```
Returns churn probability and predicted class for new customer data.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Feel free to submit issues and enhancement requests! 