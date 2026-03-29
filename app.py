import streamlit as st
import numpy as np
import pickle
import os

# Train model if not present
if not os.path.exists("model.pkl"):
    import train_model

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Title
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter your medical information to estimate diabetes risk.")

# Sidebar inputs
st.sidebar.header("Patient Information")

pregnancies = st.sidebar.slider("Pregnancies",0,15,1)
glucose = st.sidebar.slider("Glucose Level",0,200,120)
bp = st.sidebar.slider("Blood Pressure",0,150,70)
skin = st.sidebar.slider("Skin Thickness",0,100,20)
insulin = st.sidebar.slider("Insulin Level",0,900,80)
bmi = st.sidebar.slider("BMI",10.0,50.0,25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function",0.0,2.5,0.5)
age = st.sidebar.slider("Age",1,100,30)

# Predict button
if st.button("Predict Diabetes Risk"):

    data = np.array([[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f" High Risk of Diabetes ({probability*100:.1f}%)")
        st.warning("Please consult a medical professional.")
    else:
        st.success(f" Low Risk of Diabetes ({probability*100:.1f}%)")

    # Additional health info
    st.subheader("Health Recommendations")

    if prediction == 1:
        st.write("""
        • Maintain a healthy diet  
        • Exercise regularly  
        • Monitor blood glucose levels  
        • Consult a healthcare provider
        """)
    else:
        st.write("""
        • Continue maintaining a healthy lifestyle  
        • Regular exercise  
        • Balanced diet
        """)

# Footer
st.markdown("---")
st.caption("This tool is for educational purposes only.")