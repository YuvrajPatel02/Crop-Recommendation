import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Load dataset
df = pd.read_csv("kerala_crop_data.csv")  # Make sure this file is in the same folder

# Drop missing values if any
df.dropna(inplace=True)

# Define features and target
FEATURES = ["District", "Season", "Rainfall", "Temperature", "LandType", "Irrigation", "SoilType"]
TARGET = "Recommended_Crop"

# Encode categorical features
encoders = {}
for col in FEATURES + [TARGET]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders_farmer.joblib")

# Split data
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate model
labels = sorted(np.unique(y_test))
target_names = encoders[TARGET].inverse_transform(labels)
print(classification_report(y_test, model.predict(X_test), labels=labels, target_names=target_names))

# Save model
joblib.dump(model, "crop_recommender_farmer_rf.joblib")
print("✅ Model and encoders saved successfully.")
