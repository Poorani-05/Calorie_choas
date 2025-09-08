# app.py
import streamlit as st
import pandas as pd
import pickle
import altair as alt
from io import BytesIO
from fpdf import FPDF

st.set_page_config(
    page_title="🥗 Calorie Churn Prediction Advanced",
    page_icon="🍽️",
    layout="wide"
)

st.title("🥗 Advanced Calorie Churn Prediction App")
st.markdown("Predict the calorie category of food items based on nutritional information.")

# ---------------- Load model and preprocessing ----------------
with open("scaler_calorie.pkl", "rb") as f:
    scaler_data = pickle.load(f)
    scaler = scaler_data["scaler"]
    numeric_cols = scaler_data["numeric_cols"]

with open("label_encoders_calorie.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("svm_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- Tabs for UI ----------------
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------- Single Prediction ----------------
with tab1:
    st.header("Enter Nutritional Details")
    
    def user_input_features():
        data = {}
        col1, col2 = st.columns(2)
        
        # Numeric inputs via sliders
        for i, col in enumerate(numeric_cols):
            if i % 2 == 0:
                data[col] = col1.slider(col, 0.0, 500.0, 10.0, step=1.0)
            else:
                data[col] = col2.slider(col, 0.0, 500.0, 10.0, step=1.0)
        
        # Categorical inputs via selectbox
        for col in label_encoders.keys():
            data[col] = st.selectbox(col, options=list(label_encoders[col].classes_))
        
        return pd.DataFrame([data])
    
    input_df = user_input_features()
    
    # Preprocess input
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    if st.button("Predict Calorie Category"):
        pred = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        
        st.subheader("Prediction Result")
        st.success(f"Predicted Category: {pred}")
        
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({'Category': model.classes_, 'Probability': pred_proba})
        chart = alt.Chart(prob_df).mark_bar().encode(
            x='Category',
            y='Probability',
            color='Category',
            tooltip=['Category', 'Probability']
        )
        st.altair_chart(chart, use_container_width=True)
        
        # ---------------- Generate PDF report ----------------
        pdf_btn = st.button("Generate PDF Report")
        if pdf_btn:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Calorie Churn Prediction Report", ln=True, align="C")
            pdf.ln(10)
            
            pdf.set_font("Arial", "", 12)
            for col in numeric_cols:
                pdf.cell(0, 8, f"{col}: {input_df[col].values[0]:.2f}", ln=True)
            for col in label_encoders.keys():
                pdf.cell(0, 8, f"{col}: {list(label_encoders[col].classes_)[input_df[col].values[0]]}", ln=True)
            
            pdf.ln(5)
            pdf.cell(0, 8, f"Predicted Category: {pred}", ln=True)
            
            # Save PDF to bytes
            pdf_buffer = BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)
            
            st.download_button(
                "Download PDF Report",
                data=pdf_buffer,
                file_name="calorie_prediction_report.pdf",
                mime="application/pdf"
            )

# ---------------- Batch Prediction ----------------
with tab2:
    st.header("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
        
        # Encode categorical columns
        for col in batch_df.select_dtypes(include='object').columns:
            if col in label_encoders:
                batch_df[col] = label_encoders[col].transform(batch_df[col])
        
        # Scale numeric columns
        batch_df[numeric_cols] = scaler.transform(batch_df[numeric_cols])
        
        # Predict
        batch_pred = model.predict(batch_df)
        batch_pred_proba = model.predict_proba(batch_df)
        batch_df['Predicted Category'] = batch_pred
        
        st.dataframe(batch_df)
        
        st.download_button(
            label="Download Predictions CSV",
            data=batch_df.to_csv(index=False).encode('utf-8'),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("Made with ❤️ using Python, scikit-learn, CatBoost/SVM, and Streamlit")
