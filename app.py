import streamlit as st
import pandas as pd
import pickle
import altair as alt

st.title("🥗 Advanced Calorie Churn Prediction")

# Load model and preprocessing
with open("scaler_calorie.pkl", "rb") as f:
    scaler_data = pickle.load(f)
    scaler = scaler_data["scaler"]
    numeric_cols = scaler_data["numeric_cols"]

with open("label_encoders_calorie.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("svm_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input sliders
st.header("Enter Nutritional Details")
def user_input_features():
    data = {}
    for col in numeric_cols:
        data[col] = st.slider(col, min_value=0.0, max_value=500.0, step=1.0)
    for col in label_encoders.keys():
        data[col] = st.selectbox(col, options=list(label_encoders[col].classes_))
    return pd.DataFrame([data])

input_df = user_input_features()

# Preprocess
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = label_encoders[col].transform(input_df[col])

input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Prediction
if st.button("Predict Calorie Category"):
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction")
    st.write(f"**Category:** {pred}")

    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({'Category': model.classes_, 'Probability': pred_proba})
    chart = alt.Chart(prob_df).mark_bar().encode(
        x='Category',
        y='Probability',
        color='Category'
    )
    st.altair_chart(chart, use_container_width=True)
