from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    # When first opening the page, result is None
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Split date into day, month, year
    date_parts = data['date'].split('-')
    year = int(date_parts[0])
    month = int(date_parts[1])
    day = int(date_parts[2])

    # Create input data for the model
    input_data = pd.DataFrame([{
        'Crop_Type': data['crop_type'],
        'Soil_Type': data['soil_type'],
        'Soil_pH': float(data['soil_ph']),
        'Temperature': float(data['temperature']),
        'Humidity': float(data['humidity']),
        'Wind_Speed': float(data['wind_speed']),
        'N': float(data['n']),
        'P': float(data['p']),
        'K': float(data['k']),
        'Soil_Quality': data['soil_quality'],
        'year': year,
        'month': month,
        'day': day
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Ensure prediction is not negative
    prediction = max(prediction, 0)

    # Send result to template
    return render_template('index.html', result=round(float(prediction), 2))

if __name__ == '__main__':
    app.run(debug=True)
