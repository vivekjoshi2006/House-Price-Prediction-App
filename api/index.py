import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, 
            template_folder='../templates', 
            static_folder='../static')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    model = joblib.load(os.path.join(BASE_DIR, "house_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    model_cols = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))
    print("Files loaded successfully from root!")
except Exception as e:
    print(f"Error loading files: {e}")
    model_cols = []

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:          
            input_features = [float(request.form[col]) for col in model_cols]
            feature_df = pd.DataFrame([input_features], columns=model_cols)
            scaled_features = scaler.transform(feature_df)
            res = model.predict(scaled_features)[0]
            prediction = f"${res:,.2f}"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, cols=model_cols)

app = app