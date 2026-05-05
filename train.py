import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("house_data.csv")
data = data.select_dtypes(include=[np.number]).dropna()

X = data.drop("price", axis=1)
y = data["price"]

joblib.dump(list(X.columns), "model_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "house_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training Complete. 'house_model.pkl' and 'scaler.pkl' created!")