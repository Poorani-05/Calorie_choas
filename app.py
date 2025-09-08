# app.py
import streamlit as st
import pandas as pd
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
    # Numeric features
    numeric_features = [
        'Total Fat', 'Saturated Fat', 'Trans Fat', 'Cholesterol', 
        'Sodium', 'Carbohydrates', 'Dietary Fiber', 'Sugars', 'Protein'
    ]
    for col in numeric_features:
        data[col] = st.number_input(f"{col}", min_value=0.0)

    # Categorical features
    for col in ['Category', 'Serving Size']:
        if col in label_encoders:
            data[col] = st.selectbox(col, options=list(label_encoders[col].classes_))

    return pd.DataFrame([data])

input_df = user_input_features()

# ---------------- Preprocessing ----------------
# Encode categorical columns if present
for col in input_df.select_dtypes(include='object').columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

# Scale numeric columns that exist in input_df
numeric_cols_in_input = [col for col in numeric_cols if col in input_df.columns]
input_df[numeric_cols_in_input] = scaler.transform(input_df[numeric_cols_in_input])

# ---------------- Prediction ----------------
if st.button("Predict Calorie Category"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction Results")
    st.write(f"**Predicted Category:** {prediction}")
    
    st.write("**Prediction Probabilities:**")
    prob_df = pd.DataFrame([prediction_proba], columns=model.classes_)
    st.dataframe(prob_df.T.rename(columns={0: "Probability"}))
