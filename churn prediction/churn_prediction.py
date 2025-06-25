# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import sys
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# 1. Data Loading
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print('Dataset loaded successfully.')
except FileNotFoundError:
    print('File not found. Using a small dummy dataset for demonstration.')
    data = {
        'customerID': ['0001', '0002', '0003', '0004'],
        'gender': ['Female', 'Male', 'Male', 'Female'],
        'SeniorCitizen': [0, 1, 0, 0],
        'Partner': ['Yes', 'No', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes', 'Yes'],
        'tenure': [1, 34, 2, 45],
        'PhoneService': ['No', 'Yes', 'Yes', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'Yes', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No', 'Yes'],
        'TechSupport': ['No', 'No', 'Yes', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30],
        'TotalCharges': ['29.85', '1889.5', '108.15', '1840.75'],
        'Churn': ['No', 'No', 'Yes', 'No']
    }
    df = pd.DataFrame(data)

# 2. Data Cleaning and Preprocessing
print('\nInitial Data Info:')
df.info()
print('\nInitial Data Description:')
print(df.describe(include='all'))

# Handle TotalCharges: convert spaces to NaN, then to float
print('\nCleaning TotalCharges column...')
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with NaN in TotalCharges
before_drop = df.shape[0]
df = df.dropna(subset=['TotalCharges'])
after_drop = df.shape[0]
print(f'Dropped {before_drop - after_drop} rows due to missing TotalCharges.')

# Convert Churn column to 1/0
print('Converting Churn column to binary...')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features = [col for col in categorical_features if col not in ['customerID', 'Churn']]
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['Churn']]

print(f'Categorical features: {categorical_features}')
print(f'Numerical features: {numerical_features}')

# Preprocessing: One-Hot Encoding for categorical, StandardScaler for numerical
# One-Hot Encoding is preferred for nominal categorical variables (no order),
# as it avoids introducing spurious ordinal relationships (unlike Label Encoding).
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 3. Exploratory Data Analysis (Optional)
plt.figure(figsize=(5, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.close()
print("Plot saved, continuing with model training...")

# 4. Model Training
# Split data into features and target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
)

# Logistic Regression is a good starting point for binary classification due to its simplicity, interpretability, and efficiency.
logreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

logreg_pipeline.fit(X_train, y_train)

# Random Forest is robust to outliers, handles non-linearities, and often performs well with minimal tuning.
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# 5. Model Evaluation
def evaluate_model(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    print(f'\nEvaluation Metrics for {model_name}:')
    print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}')
    print(f'Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}')
    print(f'F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}')
    if y_proba is not None:
        print(f'ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}')
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Evaluate both models
evaluate_model(logreg_pipeline, X_test, y_test, 'Logistic Regression')
evaluate_model(rf_pipeline, X_test, y_test, 'Random Forest')

# 6. Prediction Function
def predict_churn(model, preprocessor, new_customer_data_dict):
    """
    Predict churn probability and class for a new customer.
    Args:
        model: Trained model pipeline (with preprocessor).
        preprocessor: The preprocessor used (for reference, not used directly here).
        new_customer_data_dict: dict of customer features (excluding customerID, Churn).
    Returns:
        churn_probability: Probability of churn (float)
        churn_class: Predicted class (0 or 1)
    """
    new_df = pd.DataFrame([new_customer_data_dict])
    # Ensure columns match training data
    # (Pipeline handles preprocessing, so just pass the DataFrame)
    churn_probability = model.predict_proba(new_df)[0][1]
    churn_class = model.predict(new_df)[0]
    return churn_probability, churn_class

# Example usage of prediction function
dummy_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

prob, pred = predict_churn(logreg_pipeline, preprocessor, dummy_customer)
print(f'\nExample Prediction (Logistic Regression): Churn Probability = {prob:.4f}, Predicted Class = {pred}')

prob_rf, pred_rf = predict_churn(rf_pipeline, preprocessor, dummy_customer)
print(f'Example Prediction (Random Forest): Churn Probability = {prob_rf:.4f}, Predicted Class = {pred_rf}')

# End of script 