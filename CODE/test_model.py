import joblib
import pickle
import numpy as np

model = joblib.load('multi_model.pkl')
le_make = pickle.load(open('le_make.pkl', 'rb'))
le_fuel = pickle.load(open('le_fuel.pkl', 'rb'))

print("Available makes:", le_make.classes_)
print("Available fuel types:", le_fuel.classes_)

# Use first valid items
make = le_make.classes_[0]
fuel_type = le_fuel.classes_[0]

engine_size = 2.0
cylinders = 4
fuel_comb = 8.5

make_encoded = le_make.transform([make])[0]
fuel_encoded = le_fuel.transform([fuel_type])[0]

input_data = np.array([[engine_size, cylinders, fuel_comb, make_encoded, fuel_encoded]])
prediction = model.predict(input_data)

co2 = round(prediction[0][0], 2)
fuel = round(prediction[0][1], 2)
efficiency = round(100 / fuel, 2) if fuel != 0 else "N/A"

print(f"Input: {engine_size}L, {cylinders} cyl, {fuel_comb} L/100km, {make}, {fuel_type}")
print(f"Predicted CO2: {co2} g/km")
print(f"Predicted Fuel: {fuel} L/100km")
print(f"Predicted Efficiency: {efficiency} km/L")
