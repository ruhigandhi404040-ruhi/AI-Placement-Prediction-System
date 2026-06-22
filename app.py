import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("placement_model.pkl", "rb"))

st.title("🎓 AI Placement Prediction System")

st.write("Enter student details below")

# Inputs
iq = st.number_input("IQ", min_value=50, max_value=200, value=100)

prev_sem = st.number_input("Previous Semester Result", min_value=0.0, max_value=10.0, value=7.0)

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)

academic = st.number_input("Academic Performance", min_value=1, max_value=10, value=7)

internship = st.selectbox(
    "Internship Experience",
    ["No", "Yes"]
)

extra = st.number_input(
    "Extra Curricular Score",
    min_value=0,
    max_value=10,
    value=5
)

communication = st.number_input(
    "Communication Skills",
    min_value=0,
    max_value=10,
    value=5
)

projects = st.number_input(
    "Projects Completed",
    min_value=0,
    max_value=20,
    value=2
)

# Convert internship to numeric
internship_value = 1 if internship == "Yes" else 0

# Prediction button
if st.button("Predict Placement"):

    input_data = pd.DataFrame([[
        iq,
        prev_sem,
        cgpa,
        academic,
        internship_value,
        extra,
        communication,
        projects
    ]],
    columns=[
        'IQ',
        'Prev_Sem_Result',
        'CGPA',
        'Academic_Performance',
        'Internship_Experience',
        'Extra_Curricular_Score',
        'Communication_Skills',
        'Projects_Completed'
    ])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ Likely to be Placed")
    else:
        st.error("❌ Placement Chances Low")

    st.write(
        f"Placement Probability: {probability[0][1]*100:.2f}%"
    )