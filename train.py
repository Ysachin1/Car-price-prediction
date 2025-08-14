import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

print("📂 Loading dataset...")
df = pd.read_csv("data/car_data.csv")

print("✅ Dataset loaded! Total rows:", len(df))

# Encode categorical features
label_encoders = {}
for col in ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split into features and target
X = df.drop("Price", axis=1)
y = df["Price"]

print("📊 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure 'model' folder exists
os.makedirs("model", exist_ok=True)

# Save model & encoders
joblib.dump(model, "model/car_price_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("💾 Model saved in 'model/' folder")
print("🎯 Training completed successfully!")
