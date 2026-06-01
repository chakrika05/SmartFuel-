import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import pickle

# import dataset
df = pd.read_csv("Data/MY2019 Fuel Consumption Ratings.csv")

# Label Encoding
le_make = LabelEncoder()
df['Make'] = le_make.fit_transform(df['Make'])

le_fuel_type = LabelEncoder()
df['Fuel_Type'] = le_fuel_type.fit_transform(df['Fuel_Type'])

# Converting Pandas DataFrame into Numpy array
X = df[['Engine_Size (L)', 'Make', 'Fuel_Type', 'Cylinders', 'Fuel_Consumption_City', 'Fuel_Consumption_Hwy']] .values
y = df[['CO2_Emissions']] .values

# linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)
multiple_linear_regression = LinearRegression(fit_intercept= True, copy_X= True, n_jobs= -1)
multiple_linear_regression.fit(X_train, y_train)

# Saving model to the disk
pickle.dump(multiple_linear_regression, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))