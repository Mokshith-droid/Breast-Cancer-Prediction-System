from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
with open("breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = []

    for i in range(1, 31):
        features.append(float(request.form[f'f{i}']))

    data = np.array(features).reshape(1, -1)

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probability = max(model.predict_proba(data)[0])
    confidence = round(probability * 100, 2)

    if prediction == 0:
        result = "Malignant Tumor Detected"
    else:
        result = "Benign Tumor Detected"

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)