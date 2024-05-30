from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('stockprices.pkl' , 'rb'))
scale = pickle.load(open('scaledvalues.pkl' , 'rb'))

@app.route('/', methods=['GET'])
def input_form():
    return render_template('input_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    open_price = float(request.form.get('open'))
    high_price = float(request.form.get('high'))
    low_price = float(request.form.get('low'))
    volume = float(request.form.get('volume'))

    # Create features
    features = np.array([open_price, high_price, low_price, volume]).reshape(1, -1)
    
    # Predict the stock price
    scaled = scale.transform(features)
    prediction = model.predict(scaled)[0]
    
    return render_template('prediction_result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)