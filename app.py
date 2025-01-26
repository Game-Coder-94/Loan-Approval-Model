import os
from flask import Flask, request, jsonify
import pickle

# Load the model
with open('loan_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['0']),
            float(request.form['1']),
            float(request.form['2']),
            float(request.form['3']),
            float(request.form['4']),
            float(request.form['5']),
            float(request.form['6']),
            float(request.form['7']),
            float(request.form['8']),
            float(request.form['9']),
            float(request.form['10']),
            float(request.form['12']),
            float(request.form['13']),
            float(request.form['14']),
            float(request.form['Clusters']),
            # Add more fields as required
        ]
        prediction = model.predict([features])
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)