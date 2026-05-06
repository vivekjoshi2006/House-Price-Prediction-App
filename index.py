import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


try:
    model = joblib.load(os.path.join(BASE_DIR, "house_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    model_cols = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading model files: {e}")
    model_cols = []  # Fallback to prevent crash


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
            prediction = f"Error in calculation: {e}"

    return render_template("index.html", prediction=prediction, cols=model_cols)


if __name__ == "__main__":
    app.run(debug=True)
