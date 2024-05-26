from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import numpy as np
import pickle

# Define FastAPI app
app = FastAPI()

# Function to load the trained model
def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

# Load the model
loaded_model = load_model('my-model.pkl')

# Prediction function
def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    # Prepare feature vector
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    # Make prediction
    prediction = loaded_model.predict(single_pred)
    return prediction.item().title()

# Define route for index page with form
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Crop Advisor: Intelligent Crop Recommendation 🌿</title>
        <style>
            body {
                background-color: cyan;
                text-align: center;
            }
            form {
                margin: auto;
                width: 50%;
                padding: 20px;
                background-color: #f2f2f2;
                border-radius: 5px;
            }
            input[type=number] {
                width: 100%;
                padding: 12px 20px;
                margin: 8px 0;
                display: inline-block;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }
            input[type=submit] {
                width: 100%;
                background-color: #4CAF50;
                color: white;
                padding: 14px 20px;
                margin: 8px 0;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <h1>Crop Advisor: Intelligent Crop Recommendation 🌿</h1>
        <form action="/predict" method="post">
            <label for="N">Nitrogen:</label>
            <input type="number" name="N" required><br>
            <label for="P">Phosphorus:</label>
            <input type="number" name="P" required><br>
            <label for="K">Potassium:</label>
            <input type="number" name="K" required><br>
            <label for="temp">Temperature:</label>
            <input type="number" name="temp" required><br>
            <label for="humidity">Humidity in %:</label>
            <input type="number" name="humidity" required><br>
            <label for="ph">pH:</label>
            <input type="number" name="ph" required><br>
            <label for="rainfall">Rainfall in mm:</label>
            <input type="number" name="rainfall" required><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """

# Define route for prediction
@app.post("/predict")
async def predict(N: float = Form(...), P: float = Form(...), K: float = Form(...), temp: float = Form(...), humidity: float = Form(...), ph: float = Form(...), rainfall: float = Form(...)):
    prediction = predict_crop(N, P, K, temp, humidity, ph, rainfall)
    return {"recommended_crop": prediction}



