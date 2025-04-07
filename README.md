# 🏠 House Price Prediction App

A Streamlit web application that predicts house prices based on various property features using a trained machine learning model.

## Features
- Interactive input form for property features
- Real-time price prediction
- Clean, user-friendly interface

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install streamlit joblib numpy pandas scikit-learn
```

## Usage
Run the application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Input Features
The model uses these 13 features for prediction:
- CRIM: Crime rate
- ZN: Residential land zoned
- INDUS: Non-retail business acres
- CHAS: Charles River dummy variable
- NOX: Nitric oxides concentration
- RM: Average rooms per dwelling
- AGE: Age of property
- DIS: Distance to employment centers
- RAD: Index of accessibility
- TAX: Property tax rate
- PTRATIO: Pupil-teacher ratio
- B: Proportion of Black population
- LSTAT: Lower status population percentage

## Model Details
- Trained model: `house_price_model.pkl`
- Feature scaler: `scaler.pkl`
- Training data: `housing.csv`

## Files
- `app.py`: Main application file
- `house_price_model.pkl`: Serialized trained model
- `scaler.pkl`: Feature scaler
- `housing.csv`: Training dataset
- `Task1.ipynb`: Jupyter notebook with model development code

## License
[MIT](https://choosealicense.com/licenses/mit/)
