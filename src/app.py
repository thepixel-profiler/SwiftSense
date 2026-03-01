import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="SwiftSense | AI Fraud Monitor", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_assets():
    """Load the trained model and column metadata with correct pathing."""
    try:
        # Use relative paths that point to the src folder
        model = joblib.load('src/swift_sense_model.pkl')
        model_cols = joblib.load('src/model_columns.pkl')
        return model, model_cols
    except FileNotFoundError:
        # Fallback for local testing if running from within the src folder
        try:
            model = joblib.load('swift_sense_model.pkl')
            model_cols = joblib.load('model_columns.pkl')
            return model, model_cols
        except FileNotFoundError:
            st.error("Model files not found in 'src/' or root. Please ensure .pkl files are uploaded to GitHub.")
            return None, None

model, model_cols = load_assets()

# --- UI HEADER ---
st.title("🛡️ SwiftSense: Real-Time Fraud Detection")
st.markdown("### Advanced Behavioral Feature Engineering Engine")
st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Transaction Parameters")

def get_user_input():
    # Basic Inputs
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=1250.0)
    old_balance = st.sidebar.number_input("Sender Original Balance", min_value=0.0, value=5000.0)
    dest_old_balance = st.sidebar.number_input("Recipient Original Balance", min_value=0.0, value=0.0)
    
    # Selecting the type (Matches PaySim categories)
    txn_type = st.sidebar.selectbox("Transaction Type", 
                                   ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
    
    # Step (representing hour of the month)
    step = st.sidebar.slider("Step (Hour of Month)", 0, 744, 150)
    
    # Simulation: Step Count (Velocity)
    step_count = st.sidebar.number_input("Transactions in this Hour (Velocity)", min_value=1, value=1)

    return {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': old_balance - amount,
        'oldbalanceDest': dest_old_balance,
        'newbalanceDest': dest_old_balance + amount,
        'type': txn_type,
        'step_count': step_count
    }

user_data = get_user_input()

# --- FEATURE ENGINEERING PIPELINE ---
def process_for_prediction(input_dict, columns):
    df = pd.DataFrame([input_dict])
    
    # 1. Logic-Based Engineered Features (Matching engine.py)
    # Balance Integrity
    df['orig_balance_err'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['dest_balance_err'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # Amount Deviation
    df['amount_diff_avg'] = df['amount'] - 150000 
    
    # 2. One-Hot Encoding
    df = pd.get_dummies(df, columns=['type'])
    
    # 3. Align with training columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0
            
    return df[columns]

# --- PREDICTION & DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.write("#### Raw Transaction Data")
    st.dataframe(pd.DataFrame([user_data]), use_container_width=True)

with col2:
    st.write("#### Risk Analysis")
    if st.button("Run SwiftSense Audit", use_container_width=True):
        if model is not None:
            # Transform input
            final_input = process_for_prediction(user_data, model_cols)
            
            # Predict
            prob = model.predict_proba(final_input)[0][1]
            prediction = model.predict(final_input)[0]
            
            # Visualization
            if prediction == 1:
                st.error(f"### ⚠️ FRAUD DETECTED (Probability: {prob:.2%})")
                st.warning("High-risk behavioral patterns identified in balance integrity and velocity.")
            else:
                st.success(f"### ✅ TRANSACTION SECURE (Probability: {prob:.2%})")
                st.balloons()
        else:
            st.error("Model engine is offline. Check if .pkl files are in the 'src' folder.")

# --- FOOTER ---
st.divider()
st.caption("Developed for IBM-NASSCOM PBEL Internship | Build: SwiftSense v1.0")