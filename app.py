from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from dep import AdvancedRiskEngine  # Replace with the actual module name

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
MODEL_PATH = "MTL.h5"
SCALER_PATH = "scaler.pkl"
engine = AdvancedRiskEngine(MODEL_PATH, SCALER_PATH)

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    patient_data = request.get_json(force=True)
    
    # Run simulation
    simulation = engine.run_simulation(patient_data)
    
    # Return simulation results
    return jsonify({
        "now_risks": simulation["now_risks"],
        "forecast_risks": simulation["forecast_risks"],
        "direct_forecast_risks": simulation["direct_forecast_risks"],
        "phenotype": simulation["phenotype"],
        "forecast_accuracy": simulation["forecast_accuracy"]
    })

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)