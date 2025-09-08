# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Calorie Churn Prediction",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 Calorie Churn Prediction App")
st.write("Predict the calorie category of food items based on nutritional information.")

# ---------------- Load preprocessing tools and model ----------------
with open("scaler_calorie.pkl", "rb") as f:
    scaler_data = pickle.load(f)
    scaler = scaler_data["scaler"]
    numeric_cols = scaler_data["numeric_cols"]

with open("label_encoders_calorie.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("svm_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- User input ----------------
st.header("Enter Nutritional Details")

def user_input_features():
    data = {}
    data['Total Fat'] = st.number_input("Total Fat (g)", min_value=0.0)
    data['Saturated Fat'] = st.number_input("Saturated Fat (g)", min_value=0.0)
    data['Trans Fat'] = st.number_input("Trans Fat (g)", min_value=0.0)
    data['Cholesterol'] = st.number_input("Cholesterol (mg)", min_value=0.0)
    data['Sodium'] = st.number_input("Sodium (mg)", min_value=0.0)
    data['Carbohydrates'] = st.number_input("Carbohydrates (g)", min_value=0.0)
    data['Dietary Fiber'] = st.number_input("Dietary Fiber (g)", min_value=0.0)
    data['Sugars'] = st.number_input("Sugars (g)", min_value=0.0)
    data['Protein'] = st.number_input("Protein (g)", min_value=0.0)
    
    data['Category'] = st.selectbox(
        "Category",
        options=list(label_encoders['Category'].classes_)
    )
    
    data['Serving Size'] = st.selectbox(
        "Serving Size",
        options=list(label_encoders['Serving Size'].classes_)
    )
    
    return pd.DataFrame([data])

input_df = user_input_features()

# ---------------- Preprocessing ----------------
# Encode categorical columns
for col in ['Category', 'Serving Size']:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ---------------- Prediction ----------------
if st.button("Predict Calorie Category"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction Results")
    st.write(f"**Predicted Category:** {prediction}")
    
    st.write("**Prediction Probabilities:**")
    prob_df = pd.DataFrame([prediction_proba], columns=model.classes_)
    st.dataframe(prob_df.T.rename(columns={0: "Probability"}))
