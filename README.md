# 🏠 House Price Predictor (ML Web App)

A professional end-to-end Machine Learning web application that predicts real estate prices. This project uses a **Random Forest Regressor** to analyze property features and provide accurate price estimations through a clean, modern user interface.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Deployment-Vercel-black.svg)](https://vercel.com/)

---

## 🚀 Key Features

*   **Advanced ML Engine:** Utilizes Random Forest Regression for superior accuracy over simple linear models.
*   **Persistent Modeling:** Implements `joblib` serialization (Pickle) to store the trained model, allowing for instant predictions without retraining.
*   **Dynamic UI:** The frontend automatically generates input fields based on the dataset features.
*   **Data Scaling:** Includes a pre-configured `StandardScaler` to ensure all numerical inputs are normalized for the AI.
*   **Fully Responsive:** Modern CSS with glass-morphism effects and Google Fonts (Poppins).

---

## 🛠️ Tech Stack

*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **Backend:** Python 3.x, Flask
*   **Persistence:** Joblib (Serialization)
*   **Frontend:** HTML5, CSS3 (Jinja2)
*   **Hosting:** Vercel

---

## 📁 Project Structure

```text
HOUSE_PRICE_APP/
├── static/
│   └── style.css          # Modern UI styling
├── templates/
│   └── index.html         # Dynamic prediction form
├── train.py               # Data preprocessing & Model training script
├── index.py               # Main Flask application (Vercel Entry Point)
├── house_data.csv         # Raw dataset
├── house_model.pkl        # Saved AI Model (Brain)
├── scaler.pkl             # Saved Data Scaler
├── model_columns.pkl      # Saved Feature Reference
├── requirements.txt       # Project dependencies
└── vercel.json            # Deployment configuration
```

---

## ⚙️ Setup & Installation

### 1. Clone & Install
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

### 2. Train the AI
Before running the website, you must generate the model files from your data:
```bash
python train.py
```
*This will create `house_model.pkl`, `scaler.pkl`, and `model_columns.pkl`.*

### 3. Run Locally
```bash
python index.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

## 🧠 How It Works (The ML Pipeline)

1.  **Preprocessing:** The `train.py` script cleans the CSV data, handles missing values, and encodes categorical variables.
2.  **Scaling:** We use `StandardScaler` to ensure features like "Total Square Feet" don't overwhelm smaller features like "Number of Bedrooms."
3.  **Training:** A **Random Forest Regressor** (an ensemble of 100 decision trees) is trained on 80% of the data.
4.  **Serialization:** The trained model and scaler are "pickled" into `.pkl` files.
5.  **Production:** The Flask app (`index.py`) loads these files and makes real-time predictions based on user input.

---

## 📄 License

Distributed under the MIT License.

**Built with ❤️ by VIVEK JOSHI**
