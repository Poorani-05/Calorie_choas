# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF  # for PDF report generation

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🥗 Calorie Category Prediction",
    page_icon="🥗",
    layout="centered"
)

# -------------------- LOAD ASSETS --------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("svm_best_model.pkl")
        scaler = joblib.load("scaler_calorie.pkl")
        label_encoders = joblib.load("label_encoders_calorie.pkl")
        return model, scaler, label_encoders
    except FileNotFoundError as e:
        st.error(f"❌ Required file not found: {e}")
        st.stop()

model, scaler, label_encoders = load_assets()

# Define columns
numeric_cols = [
    "Total Fat", "Saturated Fat", "Trans Fat", "Cholesterol",
    "Sodium", "Carbohydrates", "Dietary Fiber", "Sugars", "Protein"
]
categorical_cols = ["Category"]

# -------------------- PDF REPORT GENERATION --------------------
def generate_pdf(input_data, prediction, probabilities):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="🥗 Calorie Category Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt="📋 Input Details:", ln=True)
    for key, value in input_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt=f"🔍 Predicted Category: {prediction}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="📊 Prediction Probabilities:", ln=True)
    for cat, prob in probabilities.items():
        pdf.cell(200, 10, txt=f"{cat}: {prob:.2f}", ln=True)

    pdf.output("calorie_prediction_report.pdf")
    return "calorie_prediction_report.pdf"

# -------------------- USER INPUT FORM --------------------
def user_input_form():
    st.subheader("📥 Enter Nutritional Details")

    input_data = {}
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col} (g or mg)", min_value=0.0, step=0.1)

    input_data["Category"] = st.selectbox("Category", label_encoders["Category"].classes_)
    return pd.DataFrame([input_data])

# -------------------- MAIN APP --------------------
st.title("🥗 Calorie Category Prediction App")
st.write("Predict the calorie category of food items based on nutritional information.")

input_df = user_input_form()

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict"):
    try:
        # Encode categorical
        for col in categorical_cols:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

        # Scale numeric
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Decode label
        prediction_label = label_encoders["Calories_cat"].inverse_transform([prediction])[0]
        prob_dict = dict(zip(label_encoders["Calories_cat"].classes_, probabilities))

        st.success(f"✅ Predicted Category: **{prediction_label}**")
        st.write("📊 Prediction Probabilities:")
        st.dataframe(pd.DataFrame(prob_dict.items(), columns=["Category", "Probability"]))

        # PDF Download
        pdf_path = generate_pdf(
            {col: input_df[col].values[0] for col in input_df.columns},
            prediction_label,
            prob_dict
        )
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=pdf_path)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("🚀 Built with ❤️ using Streamlit, Scikit-Learn, and FPDF2")
