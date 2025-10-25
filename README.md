# 🧠 Chronic Kidney Disease (CKD) Prediction using Machine Learning

This project is an end-to-end **machine learning system** that predicts **Chronic Kidney Disease (CKD)** based on patient diagnostic data.  
It compares multiple models — **Logistic Regression**, **Random Forest**, **Extra Trees**, and **XGBoost** — to identify the best-performing algorithm.  
The trained models are deployed via a **Streamlit web application** for real-time predictions.

---

## 🚀 Features
- Preprocessed medical dataset with 25+ health indicators
- Compared 4 major ML models:
  - Logistic Regression  
  - Random Forest  
  - Extra Trees ✅ (best performing model)  
  - XGBoost
- Achieved **highest accuracy with Extra Trees Classifier**
- Interactive Streamlit UI for CKD prediction
- Visualized confusion matrices and performance metrics

---

## 🧩 Dataset
- **Source:** `kidney_disease.csv`
- Contains features like:
  - Blood pressure, blood glucose, serum creatinine, hemoglobin, sodium, potassium, etc.
- Missing values handled using median/mode imputation
- Categorical features label-encoded

---

## ⚙️ Model Workflow
1. Data preprocessing and cleaning  
2. Encoding categorical variables  
3. Feature scaling (for linear models)  
4. Training and evaluation across multiple classifiers  
5. Visualization of model performance  
6. Deployment using Streamlit

---

## 🧮 Model Comparison

| Model | Accuracy | Notes |
|--------|-----------|--------|
| Logistic Regression | ~97% | Fast, baseline linear model |
| Random Forest | ~99% | Strong ensemble baseline |
| **Extra Trees Classifier** | **100% ✅** | Best performer on this dataset |
| XGBoost | ~99% | High accuracy, slightly slower |

---

## 💻 Streamlit Web App
The app allows users to input patient data and instantly predict CKD presence.

### Run the app:
```bash
streamlit run streamlit_ckd_app.py
