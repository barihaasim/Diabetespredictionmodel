import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")


st.markdown("""
<style>
.main { background-color: #eaf6ff; }

.hero {
    padding: 70px;
    text-align: center;
    background: linear-gradient(to right, #0a4f70, #0f6c91);
    color: white;
    border-radius: 15px;
}

.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    margin-top: 20px;
}
.low { background: #b8e0f0; color: #043d52; }
.medium { background: #a8d5eb; color: #043d52; }
.high { background: #98cae6; color: #043d52; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="hero">
    <h1>DIABETES RISK ASSESSMENT</h1>
    <p>Diabetes affects how your body processes blood sugar.</p>
    <p>Early detection helps prevent serious complications.</p>
    </div>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists("diabetes_model.pkl"):
        return joblib.load("diabetes_model.pkl")
    return None

@st.cache_resource
def load_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    return None

model = load_model()
scaler = load_scaler()

# Inputs 
st.subheader("Enter Your Health Details")

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", 0)
    glucose = st.number_input("Glucose", 0.0)
    skin_thickness = st.number_input("Skin Thickness", 0.0)

with col2:
    blood_pressure = st.number_input("Blood Pressure", 0.0)
    bmi = st.number_input("BMI", 0.0)
    insulin = st.number_input("Insulin Level", 0.0)

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0)
    age = st.number_input("Age", 0)

# prediction
if st.button("Check Risk Level"):

    if glucose == 0 or blood_pressure == 0 or bmi == 0:
        st.error("Please enter valid health values.")
    else:
        data = np.array([[
            glucose,
            bmi,
            pregnancies,
            skin_thickness,
            dpf,
            age
        ]])

        if model and scaler:
            data_scaled = scaler.transform(data)
            prob = model.predict_proba(data_scaled)[0][1]
        else:
            prob = np.mean(data) / 200

        if prob < 0.3:
            risk = "LOW RISK"
            css = "low"
        elif prob < 0.7:
            risk = "MEDIUM RISK"
            css = "medium"
        else:
            risk = "HIGH RISK"
            css = "high"

        # Result Box
        st.markdown(f'<div class="result {css}">Risk Level: {risk}</div>', unsafe_allow_html=True)

        # Explanation
        if risk == "LOW RISK":
            st.success("Your health indicators suggest a low likelihood of diabetes.")
        elif risk == "MEDIUM RISK":
            st.markdown('<div style="background-color: #a8d5eb; padding: 15px; border-radius: 10px; color: #043d52;"><strong>Some indicators are elevated. Monitoring is recommended.</strong></div>', unsafe_allow_html=True)
        else:
            st.error("High likelihood of diabetes. Please consult a doctor.")

        # risk factors and recommendations
        factors = []
        if glucose > 140:
            factors.append("High Glucose")
        if bmi > 30:
            factors.append("High BMI")
        if age > 45:
            factors.append("Age Risk")

        if factors:
            st.write("### Key Risk Factors:")
            for f in factors:
                st.write(f"- {f}")

            st.write("### Recommendations:")

        if risk == "LOW RISK":
            st.write("- Maintain healthy lifestyle with regular exercise")
            st.write("- Eat a balanced diet rich in whole grains and vegetables")
            st.write("- Get routine health checkups annually")
        elif risk == "MEDIUM RISK":
            st.write("- Monitor blood sugar levels regularly")
            st.write("- Reduce sugar and refined carbohydrate intake")
            st.write("- Increase physical activity to at least 30 minutes daily")
        else:
            st.write("- Seek medical consultation immediately")
            st.write("- Get comprehensive diabetes screening and testing")
            st.write("- Start a supervised diet and exercise program with professional guidance")