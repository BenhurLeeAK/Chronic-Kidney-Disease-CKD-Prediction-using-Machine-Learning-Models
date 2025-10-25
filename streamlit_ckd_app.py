# streamlit_ckd_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import pickle

# -------------------------
# 1. Load Dataset & Pretrained Models
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")
    df.replace('?', pd.NA, inplace=True)
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# You can save your trained models to disk (pickle) and load here
# For demo, we will retrain Random Forest / Extra Trees / XGBoost quickly
@st.cache_resource
def train_models(df):
    # Preprocessing
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = pd.Categorical(df[col]).codes
        else:
            df[col] = df[col].fillna(df[col].median())
    
    target_col = 'classification'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Feature scaling for Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Models
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X, y)
    
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_clf.fit(X, y)
    
    return lr, rf, et, xgb_clf, scaler, X.columns.tolist()

lr_model, rf_model, et_model, xgb_model, scaler, feature_cols = train_models(df)

# -------------------------
# 2. Streamlit App
# -------------------------
st.title("Chronic Kidney Disease (CKD) Prediction")
st.write("Enter patient details below:")

with st.form("ckd_form"):
    input_data = {}
    for col in feature_cols:
        # Numeric input
        input_data[col] = st.number_input(f"{col}", value=0.0)
    
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost"])
    submitted = st.form_submit_button("Predict CKD")

if submitted:
    user_df = pd.DataFrame([input_data])
    
    # Preprocess same as training
    for col in user_df.columns:
        if user_df[col].dtype == object:
            user_df[col] = pd.Categorical(user_df[col]).codes
    
    if model_choice == "Logistic Regression":
        user_scaled = scaler.transform(user_df)
        pred = lr_model.predict(user_scaled)[0]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(user_df)[0]
    elif model_choice == "Extra Trees":
        pred = et_model.predict(user_df)[0]
    elif model_choice == "XGBoost":
        pred = xgb_model.predict(user_df)[0]
    
    # Display results
    if pred == 0:
        st.success("Prediction: No CKD detected ✅")
    else:
        st.error("Prediction: CKD detected ⚠️")
