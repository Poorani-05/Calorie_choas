# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# ---------------- Helper function to load pickle files safely ----------------
def load_pickle(file_name):
    if not os.path.exists(file_name):
        st.error(f"❌ File `{file_name}` not found. Make sure it is uploaded to your repo.")
        st.stop()
    return joblib.load(file_name)

# ---------------- Load preprocessing tools and model ----------------
scaler_data = load_pickle("scaler_calorie.pkl")
label_encoders = load_pickle("label_encoders_calorie.pkl")
model = load_pickle("svm_best_model.pkl")

scaler = scaler_data["scaler"]
numeric_cols = scaler_data["numeric_cols"]

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="🥗 Calorie Churn Prediction", page_icon="🍴", layout="wide")

st.title("🥗 Calorie Churn Prediction App")
st.markdown("Predict the **calorie category** of food items based on nutritional information.")

# Sidebar input
st.sidebar.header("📊 Enter Nutritional Details")
def user_input_features():
    data = {}
    for col in numeric_cols:
        data[col] = st.sidebar.number_input(f"{col}", min_value=0.0, step=0.1)

    # Handle categorical inputs
    for col in label_encoders.keys():
        data[col] = st.sidebar.selectbox(f"{col}", options=list(label_encoders[col].classes_))
    return pd.DataFrame([data])

input_df = user_input_features()

# ---------------- Preprocessing ----------------
input_encoded = input_df.copy()

# Encode categorical features
for col, le in label_encoders.items():
    if col in input_encoded.columns:
        input_encoded[col] = le.transform([input_encoded[col][0]])

# Scale numeric features
numeric_cols_in_input = [col for col in numeric_cols if col in input_encoded.columns]
input_encoded[numeric_cols_in_input] = scaler.transform(input_encoded[numeric_cols_in_input])

# ---------------- Prediction ----------------
prediction = model.predict(input_encoded)[0]
prediction_proba = model.predict_proba(input_encoded)[0]

# Decode prediction if it's encoded
if "Calories_cat" in label_encoders:
    prediction = label_encoders["Calories_cat"].inverse_transform([prediction])[0]

# ---------------- Results ----------------
st.subheader("🔮 Prediction Results")
st.write(f"**Predicted Category:** {prediction}")

# Probability chart
proba_df = pd.DataFrame({
    "Category": label_encoders["Calories_cat"].classes_,
    "Probability": prediction_proba
})

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x="Category", y="Probability", data=proba_df, ax=ax, palette="viridis")
plt.title("Prediction Probabilities")
st.pyplot(fig)

# ---------------- Generate PDF Report ----------------
if st.button("📄 Generate Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Calorie Churn Prediction Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Predicted Category: {prediction}", ln=True)
    pdf.cell(200, 10, "Prediction Probabilities:", ln=True)

    for i, row in proba_df.iterrows():
        pdf.cell(200, 10, f"{row['Category']}: {row['Probability']:.2f}", ln=True)

    pdf.output("calorie_prediction_report.pdf")
    with open("calorie_prediction_report.pdf", "rb") as f:
        st.download_button("⬇️ Download Report", f, file_name="calorie_prediction_report.pdf")

st.success("✅ App is running successfully!")
