from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("finance_model.pkl")  # Pre-trained financial model

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    income = data["income"]
    expenses = data["expenses"]
    
    prediction = model.predict([[income, expenses]])[0]
    advice = "Save more!" if prediction < 0 else "Good financial health!"
    
    return jsonify({"advice": advice})

if __name__ == "__main__":
    app.run(debug=True)
