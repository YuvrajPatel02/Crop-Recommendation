# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = joblib.load("crop_recommender_farmer_rf.joblib")
encoders = joblib.load("encoders_farmer.joblib")

FEATURES = ["District", "Season", "Rainfall", "Temperature", "LandType", "Irrigation", "SoilType"]

def safe_transform(col, value):
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        le.classes_ = np.append(le.classes_, value)
        return le.transform([value])[0]

@app.route('/api/recommend', methods=['POST'])
def recommend_crop():
    try:
        data = request.json
        
        # Validate input
        required_fields = FEATURES
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Encode features
        for col in FEATURES:
            df[col] = df[col].apply(lambda x: safe_transform(col, x))
        
        # Get predictions
        probs = model.predict_proba(df)[0]
        crop_names = encoders["Recommended_Crop"].inverse_transform(model.classes_)
        
        # Get top 3 recommendations
        top3_idx = probs.argsort()[-3:][::-1]
        top3_crops = [{"crop": crop_names[i], "confidence": round(probs[i] * 100, 2)} for i in top3_idx]
        
        return jsonify({
            'success': True,
            'recommendations': top3_crops,
            'input': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/options', methods=['GET'])
def get_options():
    """Get available options for each feature"""
    try:
        options = {}
        for feature in FEATURES:
            if feature in encoders:
                options[feature] = list(encoders[feature].classes_)
        
        options['Recommended_Crop'] = list(encoders['Recommended_Crop'].classes_)
        
        return jsonify(options)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Railway provides PORT via environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)