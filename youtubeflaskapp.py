from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and the feature list
try:
    model = joblib.load('model.pkl') 
    model_features = joblib.load('features.pkl')
    print("Final tuned model and feature list loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl or features.pkl not found.")
    model = None
    model_features = None

# --- Main Predictor Page ---
@app.route('/')
def home():
    """Renders the main predictor page."""
    return render_template('index.html')

# --- NEW: Dashboard Page ---
@app.route('/dashboard')
def dashboard():
    """Renders the data visualization dashboard page."""
    return render_template('dashboard.html')

# --- Prediction Logic ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if model is None or model_features is None:
        return render_template('index.html', prediction_text='Model not loaded. Please check server logs.')

    try:
        form_data = request.form.to_dict()
        for key, value in form_data.items():
            form_data[key] = float(value)

        epsilon = 1e-6
        form_data['EngagementRate'] = (form_data['New Comments'] + form_data['Likes'] + form_data['Shares']) / (form_data['Views'] + epsilon)

        features_df = pd.DataFrame([form_data])[model_features]
        prediction_log = model.predict(features_df)
        prediction = np.expm1(prediction_log)
        output = round(max(0, prediction[0]), 2)
        
        return render_template('index.html', prediction_text=f'Estimated Revenue: ${output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
