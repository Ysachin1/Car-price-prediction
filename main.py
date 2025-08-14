import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("data/car_data.csv")

# Encode categorical columns
label_encoders = {}
categorical_cols = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Target
X = df.drop(columns=["Car ID", "Price"])
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#  Print evaluation results
print("\n Model Evaluation:")
print(f"   - Root Mean Squared Error: {rmse:,.2f}")

#  Ask user for car details
print("\n Enter car details for price prediction:")
brand = input("   Brand (e.g., Tesla, Ford, BMW): ")
year = int(input("   Year: "))
engine_size = float(input("   Engine Size (in liters): "))
fuel_type = input("   Fuel Type (e.g., Petrol, Diesel, Electric, Hybrid): ")
transmission = input("   Transmission (Manual/Automatic): ")
mileage = int(input("   Mileage (in km): "))
condition = input("   Condition (e.g., New, Like New, Used): ")
model_name = input("   Model Name (e.g., Model X, Mustang): ")

# Convert user input into model-readable format
try:
    sample_input = pd.DataFrame([{
        "Brand": label_encoders["Brand"].transform([brand])[0],
        "Year": year,
        "Engine Size": engine_size,
        "Fuel Type": label_encoders["Fuel Type"].transform([fuel_type])[0],
        "Transmission": label_encoders["Transmission"].transform([transmission])[0],
        "Mileage": mileage,
        "Condition": label_encoders["Condition"].transform([condition])[0],
        "Model": label_encoders["Model"].transform([model_name])[0]
    }])

    predicted_price = model.predict(sample_input)[0]

    print("\n💡 Prediction Result:")
    print(f"   Brand: {brand}")
    print(f"   Year: {year}")
    print(f"   Estimated Price: ₹{predicted_price:,.2f}")

except ValueError as e:
    print("\n⚠ Error: One of the inputs was not recognized from the training data.")
    print("   Please ensure Brand, Fuel Type, Transmission, Condition, and Model match the dataset values.")
