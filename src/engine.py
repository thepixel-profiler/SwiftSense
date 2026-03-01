import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats
import joblib

def load_and_inspect(file_path):
    """Step 9 & 10: Load and audit the raw data."""
    print(f"--- Loading Data from {file_path} ---")
    df = pd.read_csv(file_path)
    
    # Audit check
    missing_data = df.isnull().sum().sum()
    fraud_counts = df['isFraud'].value_counts()
    fraud_pct = (fraud_counts[1] / len(df)) * 100
    
    print(f"Total Missing Values: {missing_data}")
    print(f"Fraudulent Transactions: {fraud_counts[1]} ({fraud_pct:.4f}%)")
    return df

def engineer_features(df):
    """Step 11: Transform raw data into SwiftSense behavioral signals."""
    print("--- Engineering SwiftSense Features ---")
    
    # 1. Transaction Velocity (Hourly burst detection)
    df['step_count'] = df.groupby('step')['step'].transform('count')
    
    # 2. Amount Deviation (Contextual spending analysis)
    type_avg = df.groupby('type')['amount'].transform('mean')
    df['amount_diff_avg'] = df['amount'] - type_avg
    
    # 3. Balance Integrity (Domain-specific error checking)
    df['orig_balance_err'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['dest_balance_err'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # 4. Data Sanitization
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=cols_to_drop)
    
    # 5. Categorical Encoding
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    return df

def balance_and_split(df):
    """Step 12: Address class imbalance using SMOTE."""
    print("--- Balancing Dataset with SMOTE ---")
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # Stratified split to maintain fraud ratio in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Resample only the training data
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test

def train_swift_sense(X_train, y_train):
    """Step 13: Train the XGBoost Brain."""
    print("--- Training XGBoost Model ---")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Path to your PaySim file
    RAW_PATH = "data/raw/paysim.csv"
    
    # Execute Pipeline
    raw_df = load_and_inspect(RAW_PATH)
    # Using a subset (e.g., first 200k rows) is recommended for initial local testing
    processed_df = engineer_features(raw_df.head(200000)) 
    
    X_train, X_test, y_train, y_test = balance_and_split(processed_df)
    
    # Train Model
    swift_model = train_swift_sense(X_train, y_train)
    
    # Quick Evaluation
    y_pred = swift_model.predict(X_test)
    print("\n--- SwiftSense Performance Report ---")
    print(classification_report(y_test, y_pred))



# Inside your if __name__ == "__main__": block, after training:
# Save the model
joblib.dump(swift_model, 'src/swift_sense_model.pkl')

# Save the column names to ensure app.py uses the same order
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'src/model_columns.pkl')

print("Model and columns saved successfully!")