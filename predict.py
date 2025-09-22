import joblib
import pandas as pd
import numpy as np

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

def recommend_crop(farmer_input):
    df = pd.DataFrame([farmer_input])
    for col in FEATURES:
        df[col] = df[col].apply(lambda x: safe_transform(col, x))

    probs = model.predict_proba(df)[0]
    crop_names = encoders["Recommended_Crop"].inverse_transform(model.classes_)

    top3_idx = probs.argsort()[-3:][::-1]
    top3_crops = [(crop_names[i], round(probs[i] * 100, 2)) for i in top3_idx]

    return top3_crops

if __name__ == "__main__":
    print("🌾 Kerala Crop Recommendation System 🌾\n")

    sample_input = {
        "District": input("Enter District (e.g., Idukki, Thrissur): "),
        "Season": input("Enter Season (Kharif/Rabi/Summer): "),
        "Rainfall": input("Enter Rainfall (Low/Medium/High): "),
        "Temperature": input("Enter Temperature (Cool/Moderate/Hot): "),
        "LandType": input("Enter Land Type (Lowland/Upland/Hill/Coastal): "),
        "Irrigation": input("Irrigation Available? (Yes/No): "),
        "SoilType": input("Enter Soil Type (Clay/Loam/Sandy/Laterite): ")
    }

    recommendations = recommend_crop(sample_input)

    print("\n✅ Top 3 Recommended Crops for You:")
    for crop, confidence in recommendations:
        print(f"- {crop} ({confidence}% confidence)")
