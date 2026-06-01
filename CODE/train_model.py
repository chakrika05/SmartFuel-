import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

# Load dataset
df = pd.read_csv("MY2019 Fuel Consumption.csv")

# Encode categorical variables
le_make = LabelEncoder()
df['Make'] = le_make.fit_transform(df['Make'])

le_fuel = LabelEncoder()
df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])

# Save encoders
pickle.dump(le_make, open('le_make.pkl', 'wb'))
pickle.dump(le_fuel, open('le_fuel.pkl', 'wb'))

# ✅ Features WITHOUT Fuel_Consumption
X = df[['Engine_Size (L)', 'Cylinders', 'Make', 'Fuel_Type']]

# ✅ Targets: Predict both CO2 and Fuel_Consumption_Comb
y = df[['CO2_Emissions', 'Fuel_Consumption_Comb']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'multi_model.pkl')
print("✅ Model retrained and saved correctly.")
