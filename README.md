🛡️ SwiftSense: Behavioral Fraud Detection Engine
Project Domain: Financial Cybersecurity & Machine Learning

Key Focus: Advanced Feature Engineering & Explainable AI (XAI)

📌 Project Overview
SwiftSense is an end-to-end fraud detection system built to identify anomalous financial transactions with high precision. While traditional models focus on raw transaction data, SwiftSense leverages Behavioral Feature Engineering to detect complex patterns such as "Impossible Velocity" and "Balance Integrity Violations."

🚀 Key Features
Behavioral Velocity Engine: Tracks transaction frequency within specific temporal windows to identify burst-pattern fraud.

Balance Integrity Audit: Implements domain-specific logic to detect phantom money transfers and account balance inconsistencies.

Imbalance Management: Utilizes SMOTE to address the 0.13% minority fraud class, ensuring high recall.

Real-Time Dashboard: A live Streamlit interface for auditing transactions and visualizing risk scores.

📊 Performance Metrics
Fraud Recall: 0.97 (Identifies 97% of all fraudulent transactions)

Accuracy: 1.00 (Weighted average across all classes)

Algorithm: XGBoost Classifier with Strategic Hyperparameter Tuning

🛠️ Tech Stack
Language: Python 3.x

Core Libraries: Pandas, Scikit-Learn, XGBoost, Imbalanced-Learn

Frontend: Streamlit

Deployment: Model-as-a-Service (MaaS) architecture