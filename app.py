import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
    # Get user input for 'Open', 'High', 'Low', and 'Volume'
    user_input = st.text_input("Enter Open, High, Low, Volume (comma-separated):")
    if user_input:
        open_val, high, low, volume = user_input.split(',')

        # Load the model
        the_best_model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Prepare user input features
        user_input_features = np.array([float(open_val), float(high), float(low), float(volume)]).reshape(1, -1)

        # Scale user input features
        user_input_features = scaler.transform(user_input_features)

        # Make a prediction
        predicted_close = the_best_model.predict(user_input_features)

        # Print the predicted close price
        st.write(f"The predicted close price is: {predicted_close[0]:.2f}")

def app():
    st.title("Stock Price Predictor")

    main()

if __name__ == "__main__":
    app()

