import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import joblib
import os

# Load dataset
df = pd.read_csv('data/vehicles.csv', low_memory=False)

# Strip whitespaces from column names just in case
df.columns = df.columns.str.strip()

# Select required columns
df = df[['displ', 'cylinders', 'make', 'fuelType', 'trany', 'city08']]

# Rename columns for better readability
df.rename(columns={
    'displ': 'Engine_Size',
    'cylinders': 'Cylinders',
    'make': 'Make',
    'fuelType': 'Fuel_Type',
    'trany': 'Transmission',
    'city08': 'Fuel_Consumption'
}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Convert categorical features to category dtype
categorical_cols = ['Make', 'Fuel_Type', 'Transmission']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Features and target
X = df[['Engine_Size', 'Cylinders', 'Make', 'Fuel_Type', 'Transmission']]
y = df['Fuel_Consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_cols,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# Save the model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/catboost_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")
