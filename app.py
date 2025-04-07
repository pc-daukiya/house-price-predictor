import streamlit as st 
import joblib 
import numpy as np 

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏠 House Price Prediction App")
st.markdown("Enter the details below to predict the house price.")

CRIM = st.number_input("Crime Rate (CRIM)", min_value=0.0, step=0.00001, format="%.5f")
ZN = st.number_input("Residential Land Zoned (ZN)", min_value=0.0, step=0.01, format="%.2f")
INDUS = st.number_input("Non-retail Business Acres (INDUS)", min_value=0.0, step=0.01, format="%.2f")
CHAS = st.selectbox("Charles River (CHAS)", [0, 1])
NOX = st.number_input("Nitric Oxides Concentration (NOX)", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
RM = st.number_input("Avg Rooms per Dwelling (RM)", min_value=0.0, step=0.001, format="%.3f")
AGE = st.number_input("Age of Property (AGE)", min_value=0.0, step=0.1, format="%.1f")
DIS = st.number_input("Distance to Employment Centers (DIS)", min_value=0.0, step=0.0001, format="%.4f")
RAD = st.number_input("Index of Accessibility (RAD)", min_value=0.0, step=1.0, format="%.0f")
TAX = st.number_input("Property Tax Rate (TAX)", min_value=0.0, step=1.0, format="%.0f")
PTRATIO = st.number_input("Pupil-Teacher Ratio (PTRATIO)", min_value=0.0, step=0.1, format="%.1f")
B = st.number_input("Proportion of Black Population (B)", min_value=0.0, step=0.01, format="%.2f")
LSTAT = st.number_input("Lower Status Population (%) (LSTAT)", min_value=0.0, step=0.01, format="%.2f")

if st.button("Predict"):
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    input_df = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]], columns=columns)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted House Price (MEDV): ${prediction[0]*1000:.2f}")
