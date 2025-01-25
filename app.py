from flask import Flask, request, jsonify
import pickle

# Load the model
with open('loan_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form-data from the request
        features = [
            float(request.form['A15']),
            float(request.form['A14']),
            float(request.form['A13']),
            float(request.form['A12']),
            float(request.form['A11']),
            float(request.form['A10']),
            float(request.form['A9']),
            float(request.form['A8']),
            float(request.form['A7']),
            float(request.form['A6']),
            float(request.form['A5']),
            float(request.form['A4']),
            float(request.form['A3']),
            float(request.form['A2']),
            float(request.form['A1'])
        ]
        
        # Make prediction
        prediction = model.predict([features])  # Adjust based on your model's input format
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
