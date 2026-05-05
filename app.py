from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load("house_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_cols = joblib.load("model_columns.pkl")
except:
    print("Error: Model files not found. Run train.py first!")

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