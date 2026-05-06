import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'), 
            static_folder=os.path.join(BASE_DIR, 'static'))

def get_model_path(filename):
    return os.path.join(BASE_DIR, 'models', filename)

try:
    model = joblib.load(get_model_path("house_model.pkl"))
    scaler = joblib.load(get_model_path("scaler.pkl"))
    model_cols = joblib.load(get_model_path("model_columns.pkl"))
    print("AI Models loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR loading model files: {e}")
    model = None 
    model_cols = []

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        if model is None:
            return "Error: Model not loaded on server. Check if .pkl files are uploaded."
        try:          
            input_values = []
            for col in model_cols:
                val = request.form.get(col, 0)
                input_values.append(float(val))
            
            feature_df = pd.DataFrame([input_values], columns=model_cols)
            scaled_features = scaler.transform(feature_df)
            raw_prediction = model.predict(scaled_features)[0]
            prediction = f"${raw_prediction:,.2f}"
            
        except Exception as e:
            prediction = f"Error in calculation: {e}"

    return render_template("templates.html", prediction=prediction, cols=model_cols)

app = app
