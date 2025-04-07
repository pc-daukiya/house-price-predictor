import streamlit as st
import pickle
import pandas as pd

# Load the trained model and scaler
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('House Price Predictor')

# Input fields
st.header('Enter House Details')
bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Bathrooms', min_value=1, max_value=5, value=2)
sqft_living = st.number_input('Living Area (sqft)', min_value=500, max_value=10000, value=1500)
sqft_lot = st.number_input('Lot Size (sqft)', min_value=500, max_value=100000, value=5000)
floors = st.number_input('Floors', min_value=1, max_value=4, value=2)
waterfront = st.selectbox('Waterfront', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
view = st.slider('View Rating', 0, 4, 2)
condition = st.slider('Condition', 1, 5, 3)
grade = st.slider('Grade', 1, 13, 7)

# Prediction button
if st.button('Predict Price'):
    # Prepare input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, 
                              waterfront, view, condition, grade]],
                            columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                                    'floors', 'waterfront', 'view', 'condition', 'grade'])
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    
    # Display result
    st.success(f'Predicted House Price: ${prediction:,.2f}')

st.markdown("""
**Note:** This app uses the machine learning model trained in this repository.
""")
