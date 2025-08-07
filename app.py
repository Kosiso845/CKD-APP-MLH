import streamlit as st
import pandas as pd
import joblib

# ---- CONFIGURATION ----
st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# ---- CUSTOM DARK THEME CSS STYLING ----
custom_css = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00FFAA;
    }
    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        font-weight: 600;
        color: #f0f0f0;
    }
    .stButton button {
        background-color: #00FFAA;
        color: black;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #004d40;
        color: white;
    }
    .css-18e3th9 {
        background-color: #0e1117;
    }
    #footer {
        text-align: center;
        font-size: 13px;
        color: #999;
        margin-top: 40px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 style='text-align: center;'>🩺 CKD Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Quickly assess potential kidney function risk using 10 key indicators.")
st.write("---")

# ---- LOAD MODEL ----
try:
    model, feature_columns = joblib.load("ckd_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'ckd_model.pkl' not found. Make sure it's in the same directory.")
    st.stop()

# ---- USER FORM ----
with st.form("ckd_form"):
    st.markdown("#### Enter your clinical information:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
        smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
        alcohol = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
    with col2:
        systolic = st.number_input("Systolic BP", 80, 200, 120)
        diastolic = st.number_input("Diastolic BP", 50, 130, 80)
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.1)
        gfr = st.number_input("GFR (Glomerular Filtration Rate)", 5, 150, 90)
        fatigue = st.slider("Fatigue Level (1–10)", 1, 10, 5)
        family_kd = st.selectbox("Family history of kidney disease?", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict CKD Stage")

# ---- PREDICTION ----
if submitted:
    with st.spinner("⏳ Making prediction..."):
        mapper = {"No": 0, "Yes": 1, "Male": 1, "Female": 0}
        input_data = {col: 0 for col in feature_columns}

        input_data.update({
            "age": age,
            "gender": mapper[gender],
            "bmi": bmi,
            "smoking": mapper[smoking],
            "alcoholconsumption": mapper[alcohol],
            "systolicbp": systolic,
            "diastolicbp": diastolic,
            "serumcreatinine": creatinine,
            "gfr": gfr,
            "fatiguelevels": fatigue,
            "familyhistorykidneydisease": mapper[family_kd],
        })

        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]

    st.success(f"🧾 Predicted GFR Stage: **{prediction}**")
    st.markdown("💡 _This prediction is not a diagnosis. Always consult a medical professional._")

    st.download_button(
        label="📥 Download Prediction Report",
        data=input_df.to_csv(index=False),
        file_name="ckd_prediction.csv",
        mime="text/csv"
    )

# ---- FOOTER ----
footer_html = """
<div id="footer">
    Built with ❤️ by <strong>Nwaonyeoma Kosiso Jennifer</strong> | Powered by Streamlit
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
