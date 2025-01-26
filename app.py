import os
from flask import Flask, request, jsonify
import pickle

# Load the model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            # Add more fields as required
        ]
        prediction = model.predict([features])
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
