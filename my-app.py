import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

# Ignore warnings related to unpickling
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# Function to load the trained model
def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

# Main function to define Streamlit app
def main():
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: cyan;  /* Set background color to cyan */
        }
        .stButton button {
            background-color: #4CAF50;  /* Set predict button color to green */
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set title and subheader
    st.title("CropMaster: Your Farm's Best Friend 🌿")
    st.write("---")
    st.subheader("Get Personalized Crop Recommendations for Your Farm's Success 🚀")

    # Create form for user input
    with st.form("user_input_form"):
        # Layout columns for input fields
        col1, col2 = st.columns([1, 1])

        # Input fields for fertilizer levels
        with col1:
            N = st.number_input("Nitrogen", 1, 10000)
            P = st.number_input("Phosphorus", 1, 10000)
            K = st.number_input("Potassium", 1, 10000)

        # Input fields for environmental factors
        with col2:
            temp = st.number_input("Temperature", 0.0, 100000.0)
            humidity = st.number_input("Humidity in %", 0.0, 100000.0)
            ph = st.number_input("pH", 0.0, 100000.0)
            rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

        # Submit button
        submit_button = st.form_submit_button(label="Predict")

    # Perform prediction on form submission
    if submit_button:
        # Prepare feature vector
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Load model and make prediction
        loaded_model = load_model('my-model.pkl')
        prediction = loaded_model.predict(single_pred)

        # Display prediction result
        st.write('---')
        st.success(f"The recommended crop for your farm is: {prediction.item().title()}")

# Run the main function
if __name__ == '__main__':
    main()