import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

print("GUI script started...")

# Load trained model
try:
    model = joblib.load("model/car_price_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Load label encoders
try:
    label_encoders = joblib.load("model/label_encoders.pkl")
    print("✅ Label encoders loaded!")
except Exception as e:
    print("❌ Error loading label encoders:", e)
    label_encoders = {}

# Predict price
def predict_price():
    try:
        if model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        brand = brand_entry.get().strip()
        fuel = fuel_entry.get().strip()
        transmission = transmission_entry.get().strip()
        condition = condition_entry.get().strip()
        model_name = model_entry.get().strip()
        year = int(year_entry.get().strip())
        engine_size = float(engine_entry.get().strip().replace(",", "."))
        mileage = float(mileage_entry.get().strip().replace(",", "."))

        # Encode categorical features
        try:
            brand_encoded = label_encoders["Brand"].transform([brand])[0]
            fuel_encoded = label_encoders["Fuel Type"].transform([fuel])[0]
            transmission_encoded = label_encoders["Transmission"].transform([transmission])[0]
            condition_encoded = label_encoders["Condition"].transform([condition])[0]
            model_encoded = label_encoders["Model"].transform([model_name])[0]
        except:
            messagebox.showerror("Error", "One of the categorical inputs is invalid or not in training data!")
            return

        features = np.array([[brand_encoded, year, engine_size, fuel_encoded, transmission_encoded, mileage, condition_encoded, model_encoded]])
        price = model.predict(features)[0]

        messagebox.showinfo("Prediction", f"Estimated Price: ₹{price:,.2f}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values for year, engine size, and mileage.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create main window
root = tk.Tk()
root.title("Car Price Prediction")
root.geometry("400x450")

# Inputs
tk.Label(root, text="Brand Name:").pack()
brand_entry = tk.Entry(root)
brand_entry.pack()

tk.Label(root, text="Fuel Type:").pack()
fuel_entry = tk.Entry(root)
fuel_entry.pack()

tk.Label(root, text="Transmission:").pack()
transmission_entry = tk.Entry(root)
transmission_entry.pack()

tk.Label(root, text="Condition:").pack()
condition_entry = tk.Entry(root)
condition_entry.pack()

tk.Label(root, text="Model Name:").pack()
model_entry = tk.Entry(root)
model_entry.pack()

tk.Label(root, text="Year:").pack()
year_entry = tk.Entry(root)
year_entry.pack()

tk.Label(root, text="Engine Size (L):").pack()
engine_entry = tk.Entry(root)
engine_entry.pack()

tk.Label(root, text="Mileage (kmpl):").pack()
mileage_entry = tk.Entry(root)
mileage_entry.pack()

# Predict Button
tk.Button(root, text="Predict Price", command=predict_price).pack(pady=15)

print("Starting mainloop...")
root.mainloop()
