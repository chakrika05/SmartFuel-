import pandas as pd

# Load cleaned dataset (same as used in model)
df = pd.read_csv("data/vehicles.csv")
df = df[['displ', 'cylinders', 'make', 'fuelType', 'trany', 'city08']]
df.rename(columns={
    'displ': 'Engine_Size',
    'cylinders': 'Cylinders',
    'make': 'Make',
    'fuelType': 'Fuel_Type',
    'trany': 'Transmission',
    'city08': 'Fuel_Consumption'
}, inplace=True)
df.dropna(inplace=True)

# --- 👤 Simulated User Input ---
user_input = {
    "Make": "Toyota",
    "Cylinders": 4,
    "Engine_Size": 1.8,
    "fuel_pred": 7.5  # ← This is the model output (prediction)
}

# --- 🔍 Compare with Similar Cars ---
def compare_car(user_input, df):
    similar = df[
        (df['Cylinders'] == user_input['Cylinders']) &
        (df['Engine_Size'].between(user_input['Engine_Size'] - 0.5, user_input['Engine_Size'] + 0.5))
    ]

    avg_consumption = similar['Fuel_Consumption'].mean()
    user_val = user_input['fuel_pred']
    diff = user_val - avg_consumption

    print(f"\n🚗 Comparison Result:")
    if diff > 0:
        print(f"Your car uses {diff:.2f} L/100km **more** than average similar cars.")
    else:
        print(f"Your car uses {-diff:.2f} L/100km **less** than average similar cars.")

    # Recommend more efficient cars
    better = similar[similar['Fuel_Consumption'] < user_val].sort_values('Fuel_Consumption').head(5)
    print("\n✅ Suggested Greener Alternatives:")
    if better.empty:
        print("No greener cars found in similar specs.")
    else:
        print(better[['Make', 'Engine_Size', 'Cylinders', 'Fuel_Consumption']].to_string(index=False))

# --- 🔁 Run the comparison ---
compare_car(user_input, df)
