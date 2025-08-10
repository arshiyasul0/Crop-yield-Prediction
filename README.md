# 🌾 Crop Yield Prediction Web App

A Flask-based web application that predicts crop yield using **CatBoost**, **XGBoost**, and **LSTM** machine learning models.  
The app allows users to input agricultural parameters and instantly get predictions from all three models.

---

## 📌 Features

- Predict crop yield with:
  - **CatBoost Regressor**
  - **XGBoost Regressor**
  - **LSTM Neural Network**
- Data preprocessing using:
  - **StandardScaler** for normalization
  - **PCA** for dimensionality reduction
- Simple and clean HTML interface
- Multiple model results displayed side-by-side

---

## 📂 Project Structure

crop-yield-prediction-app
├── app.py
├── models/
│ ├── best_model_catboost.cbm
│ ├── best_model_xgboost.pkl
│ ├── best_model_lstm.keras
│ ├── scaler.pkl
│ └── pca.pkl
├── templates/
│ └── index.html
├── static/
└── README.md


---

## 🛠 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/arshiyasul0/crop-yield-prediction-app.git
   cd crop-yield-prediction-app

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the App**
   ```bash
   python app.py

5. **Access in Browser**
   
     http://127.0.0.1:5000/

## 📦 Requirements
   - Flask
   - catboost
   - xgboost
   - tensorflow
   - numpy
   - joblib
     
## 📄 Generate requirements.txt
   ```bash
   pip freeze > requirements.txt

   ```

## 📝 Usage
 1. Enter all required values:

    - year
    - location
    - area
    - rainfall
    - temperature
    - soil_type
    - irrigation
    - humidity
    - crops
    - season

2. Click Predict.
3. View predictions from CatBoost, XGBoost, and LSTM models.
 
 
