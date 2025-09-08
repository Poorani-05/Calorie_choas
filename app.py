# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------- Load saved artifacts ----------------
with open("svm_best_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler_data = pickle.load(f)
    scaler = scaler_data["scaler"]
    numeric_cols = scaler_data["numeric_cols"]

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Calorie Churn Prediction", page_icon="🥗")

# ---------------- App title ----------------
st.title("🥗 Calorie Churn Prediction")
st.write(
    "Enter the nutritional details of a food item and predict its calorie category (Low, Medium, High, Very High)."
)

# ---------------- User input ----------------
def user_input_features():
    input_data = {}
    # Example for numeric columns
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col}", value=0)

    # Example for categorical columns
    for col in label_encoders:
        options = label_encoders[col].classes_
        input_data[col] = st.selectbox(f"{col}", options)

    return pd.DataFrame([input_data])

input_df = user_input_features()

# ---------------- Preprocessing ----------------
# Encode categorical columns
for col in label_encoders:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ---------------- Prediction ----------------
if st.button("Predict Calorie Category"):
    prediction = svm_model.predict(input_df)[0]
    prediction_proba = svm_model.predict_proba(input_df)[0]

    # Reverse transform to get original category labels if needed
    st.subheader("Prediction Result")
    st.write(f"Predicted Calorie Category: **{prediction}**")
    
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame(
        [prediction_proba],
        columns=svm_model.classes_
    )
    st.write(prob_df.T.rename(columns={0: "Probability"}))
