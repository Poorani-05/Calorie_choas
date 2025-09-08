# app.py
import streamlit as st
import pandas as pd
import pickle
import altair as alt

st.set_page_config(
    page_title="🥗 Calorie Churn Prediction",
    page_icon="🍽️",
    layout="wide"
)

st.title("🥗 Advanced Calorie Churn Prediction App")
st.write("Predict the calorie category of food items based on nutritional information.")

# ---------------- Load model and preprocessing ----------------
with open("scaler_calorie.pkl", "rb") as f:
    scaler_data = pickle.load(f)
    scaler = scaler_data["scaler"]
    numeric_cols = scaler_data["numeric_cols"]

with open("label_encoders_calorie.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("svm_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- Sidebar for batch CSV upload ----------------
st.sidebar.header("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file for batch prediction", type=["csv"])
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV loaded successfully!")
    
    # Encode categorical columns
    for col in batch_df.select_dtypes(include='object').columns:
        if col in label_encoders:
            batch_df[col] = label_encoders[col].transform(batch_df[col])
    
    # Scale numeric columns
    batch_df[numeric_cols] = scaler.transform(batch_df[numeric_cols])
    
    # Predict
    batch_pred = model.predict(batch_df)
    batch_pred_proba = model.predict_proba(batch_df)
    
    result_df = batch_df.copy()
    result_df['Predicted Category'] = batch_pred
    st.sidebar.download_button(
        label="Download Predictions CSV",
        data=result_df.to_csv(index=False).encode('utf-8'),
        file_name="batch_predictions.csv",
        mime="text/csv"
    )

# ---------------- Single Input Prediction ----------------
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

# ---------------- Preprocess input ----------------
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = label_encoders[col].transform(input_df[col])

input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ---------------- Prediction ----------------
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

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Python, scikit-learn, and Streamlit")
