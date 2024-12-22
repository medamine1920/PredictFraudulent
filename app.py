from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("predictFraud.pkl")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/form')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = {
        'type': request.form['type'],
        'amount': float(request.form['amount']),
        'oldbalanceOrg': float(request.form['oldbalanceOrg']),
        'newbalanceOrig': float(request.form['newbalanceOrig'])
    }
    
    # Convert the form data to DataFrame
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Render prediction result
    return render_template('result.html', prediction=prediction)

@app.route('/visualizations')
def visualizations():
    # Render visualization page
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)
