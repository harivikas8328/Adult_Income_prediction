
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("adult_income_pipeline.pkl")

st.title("Adult Income Prediction")

# Input widgets
age = st.number_input("Age", 17, 100, 30)

workclass = st.selectbox(
    "Workclass",
    ["Private", "Self-emp-not-inc", "Self-emp-inc",
     "Federal-gov", "Local-gov", "State-gov"]
)

fnlwgt = st.number_input("Final Weight", min_value=1, value=100000)

education = st.selectbox(
    "Education",
    ["Bachelors", "Masters", "HS-grad",
     "Some-college", "Doctorate", "11th",
     "Assoc-voc", "Assoc-acdm"]
)

education_num = st.number_input("Education Number", 1, 16, 10)

marital_status = st.selectbox(
    "Marital Status",
    ["Married-civ-spouse", "Never-married",
     "Divorced", "Separated", "Widowed"]
)

occupation = st.selectbox(
    "Occupation",
    ["Tech-support", "Craft-repair", "Other-service",
     "Sales", "Exec-managerial", "Prof-specialty",
     "Adm-clerical", "Machine-op-inspct"]
)

relationship = st.selectbox(
    "Relationship",
    ["Wife", "Own-child", "Husband",
     "Not-in-family", "Other-relative", "Unmarried"]
)

race = st.selectbox(
    "Race",
    ["White", "Black", "Asian-Pac-Islander",
     "Amer-Indian-Eskimo", "Other"]
)

sex = st.selectbox("Sex", ["Male", "Female"])

capital_gain = st.number_input("Capital Gain", min_value=0)

capital_loss = st.number_input("Capital Loss", min_value=0)

hours = st.number_input("Hours Per Week", 1, 100, 40)

country = st.selectbox(
    "Native Country",
    ["United-States", "India", "Mexico",
     "Philippines", "Germany", "Canada"]
)

if st.button("Predict Income"):

    data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours,
        "native_country": country
    }])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Income: Above 50K")
    else:
        st.success("Income: Below or Equal to 50K")
