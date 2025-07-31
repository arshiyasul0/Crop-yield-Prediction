from flask import Flask, render_template, request
import joblib
import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load all models and preprocessing tools
catboost_model = CatBoostRegressor()
catboost_model.load_model("models/best_model_catboost.cbm")
xgb_model = joblib.load("models/best_model_xgboost.pkl")
lstm_model = load_model("models/best_model_lstm.keras")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_cat = prediction_xgb = prediction_lstm = None
    if request.method == 'POST':
        try:
            values = [float(request.form.get(f)) for f in [
                'year', 'location', 'area', 'rainfall', 'temperature', 
                'soil_type', 'irrigation', 'humidity', 'crops', 'price', 'season']]
            scaled = scaler.transform([values])
            reduced = pca.transform(scaled)
            prediction_cat = catboost_model.predict(reduced)[0]
            prediction_xgb = xgb_model.predict(reduced)[0]
            prediction_lstm = lstm_model.predict(reduced)[0][0]
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html', cat=prediction_cat, xgb=prediction_xgb, lstm=prediction_lstm)

if __name__ == '__main__':
    app.run(debug=True)
