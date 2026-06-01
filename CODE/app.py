from flask import Flask, request, render_template
import joblib
import pickle
import numpy as np
import pandas as pd
import random
from flask import request, flash
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

# Load model and encoders
model = joblib.load('multi_model.pkl')
le_make = pickle.load(open('le_make.pkl', 'rb'))
le_fuel = pickle.load(open('le_fuel.pkl', 'rb'))

make_options = le_make.classes_.tolist()
fuel_options = le_fuel.classes_.tolist()

eco_tips = [
    "Maintain correct tire pressure to reduce drag.",
    "Avoid rapid acceleration and hard braking.",
    "Remove excess weight from your car.",
    "Use cruise control on highways to save fuel.",
    "Limit air conditioner usage when possible.",
]

def compare_and_suggest(user_input, df):
    similar = df[
        (df['Cylinders'] == user_input['Cylinders']) &
        (df['Engine_Size'].between(user_input['Engine_Size'] - 0.5, user_input['Engine_Size'] + 0.5))
    ]

    avg_consumption = similar['Fuel_Consumption'].mean()
    user_val = user_input['fuel_pred']

    if not pd.isna(avg_consumption) and avg_consumption > 0:
        fuel_diff = user_val - avg_consumption
        fuel_diff_percent = round((fuel_diff / avg_consumption) * 100, 1)
        fuel_comment = "higher" if fuel_diff > 0 else "lower"
    else:
        fuel_diff = 0
        fuel_diff_percent = 0
        fuel_comment = "N/A"

    co2_diff_percent = fuel_diff_percent
    co2_comment = fuel_comment

    comparison_result = {
        "diff": round(fuel_diff, 2),
        "average": round(avg_consumption, 2) if not pd.isna(avg_consumption) else None,
        "fuel_diff_percent": abs(fuel_diff_percent),
        "fuel_comment": fuel_comment,
        "co2_diff_percent": abs(co2_diff_percent),
        "co2_comment": co2_comment,
        "better_cars": []
    }

    better = similar[similar['Fuel_Consumption'] < user_val].sort_values('Fuel_Consumption').head(5)
    if not better.empty:
        better_cars = []
        for _, row in better.iterrows():
            better_cars.append({
                'Make': row['Make'],
                'Engine_Size': row['Engine_Size'],
                'Cylinders': int(row['Cylinders']),
                'Fuel_Consumption': round(row['Fuel_Consumption'], 2),
                'CO2': round(row['Fuel_Consumption'] * 23.2, 2)
            })
        comparison_result["better_cars"] = better_cars

    return comparison_result

@app.route('/')
def index():
    return render_template('index.html', makes=make_options, fuels=fuel_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        engine_size = float(request.form['engine_size'])
        cylinders = int(request.form['cylinders'])
        make = request.form['make']
        fuel_type = request.form['fuel_type']
        price_per_litre = float(request.form.get('price_per_litre', 0))
        distance_per_month = float(request.form.get('distance_per_month', 0))

        make_encoded = le_make.transform([make])[0]
        fuel_encoded = le_fuel.transform([fuel_type])[0]

        input_data = np.array([[engine_size, cylinders, make_encoded, fuel_encoded]])
        prediction = model.predict(input_data)

        co2_pred = round(prediction[0][0], 2)
        fuel_pred = round(prediction[0][1], 2)
        km_per_liter = round(100 / fuel_pred, 2) if fuel_pred != 0 else 'N/A'

        fuel_cost = round(fuel_pred * (distance_per_month / 100) * price_per_litre, 2)
        co2_yearly_tons = round((co2_pred * distance_per_month * 12) / 1000000, 2)

        df = pd.read_csv('data/vehicles.csv')
        df = df[['displ', 'cylinders', 'make', 'city08']].dropna()
        df.columns = ['Engine_Size', 'Cylinders', 'Make', 'Fuel_Consumption']

        user_input = {
            'Make': make,
            'Cylinders': cylinders,
            'Engine_Size': engine_size,
            'fuel_pred': fuel_pred
        }

        comparison = compare_and_suggest(user_input, df)
        better_car = comparison["better_cars"][0] if comparison["better_cars"] else None

        tip = random.choice(eco_tips)

        return render_template('index.html',
                               co2=co2_pred,
                               fuel=fuel_pred,
                               efficiency=km_per_liter,
                               makes=make_options,
                               fuels=fuel_options,
                               comparison=comparison,
                               better_car=better_car,
                               tip=tip,
                               fuel_cost=fuel_cost,
                               co2_yearly_tons=co2_yearly_tons)

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/best-cars")
def best_cars():
    return render_template("best_cars.html")

@app.route('/fuel-efficient')
def fuel_efficient():
    return render_template("fuel_efficient.html")

@app.route("/sports-cars")
def sports_cars():
    return render_template("sports_cars.html")

@app.route("/family-cars")
def family_cars():
    return render_template("family_cars.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/contact", methods=["POST"])
def contact():
    name = request.form["name"]
    email = request.form["email"]
    message = request.form["message"]

    # Email setup (Use your credentials carefully)
    sender_email = "your@gmail.com"
    receiver_email = "your@gmail.com"
    password = "your_app_password"  # Use an app password (not your main Gmail password)

    subject = f"Query from {name} ({email})"
    body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)

        flash("Your message has been sent successfully!", "success")
    except Exception as e:
        print(e)
        flash("Something went wrong. Please try again.", "danger")

    return redirect(url_for("about"))
if __name__ == "__main__":
    app.run(debug=True)