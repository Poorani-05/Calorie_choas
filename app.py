# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

try:
    from fpdf import FPDF
except ImportError:
    st.error("⚠️ fpdf2 is missing! Add `fpdf2` to requirements.txt.")
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🥗 Calorie Prediction", page_icon="🥗", layout="centered")

# ---------------- LOAD ASSETS ----------------
@st.cache_resource
def load_pickle(file_name):
    if not os.path.exists(file_name):
        st.error(f"❌ File `{file_name}` not found. Upload it to your repo.")
        st.stop()
    return joblib.load(file_name)

try:
    model = load_pickle("svm_best_model.pkl")
    scaler = load_pickle("scaler_calorie.pkl")
    label_encoders = load_pickle("label_encoders_calorie.pkl")
except Exception as e:
    st.error(f"Failed to load assets: {e}")
    st.stop()

# ---------------- CONSTANTS ----------------
numeric_cols = [
    "Total Fat", "Saturated Fat", "Trans Fat", "Cholesterol",
    "Sodium", "Carbohydrates", "Dietary Fiber", "Sugars", "Protein"
]
categorical_cols = ["Category"]

# ---------------- PDF GENERATION ----------------
def generate_pdf(data, prediction, probs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "🥗 Calorie Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for k, v in data.items():
        pdf.cell(200, 10, f"{k}: {v}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, f"Prediction: {prediction}", ln=True)
    for k, v in probs.items():
        pdf.cell(200, 10, f"{k}: {v:.2f}", ln=True)
    path = "report.pdf"
    pdf.output(path)
    return path

# ---------------- INPUT FORM ----------------
def user_form():
    st.subheader("📥 Enter Nutritional Details")
    row = {}
    for col in numeric_cols:
        row[col] = st.number_input(col, min_value=0.0, step=0.1)
    row["Category"] = st.selectbox("Category", label_encoders["Category"].classes_)
    return pd.DataFrame([row])

st.title("🥗 Calorie Category Prediction")
df_input = user_form()

if st.button("🔮 Predict"):
    try:
        # Encode categorical
        for col in categorical_cols:
            df_input[col] = label_encoders[col].transform(df_input[col])

        # Scale numeric
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        # Predict
        pred = model.predict(df_input)[0]
        probs = model.predict_proba(df_input)[0]
        pred_label = label_encoders["Calories_cat"].inverse_transform([pred])[0]
        prob_dict = dict(zip(label_encoders["Calories_cat"].classes_, probs))

        # Show
        st.success(f"✅ Predicted Category: **{pred_label}**")
        st.dataframe(pd.DataFrame(prob_dict.items(), columns=["Category", "Probability"]))

        # PDF Download
        pdf_path = generate_pdf({c: df_input[c].values[0] for c in df_input.columns}, pred_label, prob_dict)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="calorie_prediction.pdf")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")

st.markdown("---")
st.caption("🚀 Built with Streamlit + Scikit-Learn + fpdf2")
