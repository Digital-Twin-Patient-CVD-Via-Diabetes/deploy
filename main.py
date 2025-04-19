from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pickle
from pgmpy.inference import VariableElimination
import os
import uvicorn

app = FastAPI(title="Bayesian Health Risk Predictor")

# Global model variable
model = None

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    model_path = "bayesian_network_modelss.pkl"
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        model = None
        return
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully. Model type: {type(model)}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        model = None

# Input schema
class HealthInput(BaseModel):
    Blood_Pressure: float
    Age: int
    Exercise_Hours_Per_Week: float
    Diet: str  # "Healthy" or "Unhealthy"
    Sleep_Hours_Per_Day: float
    Stress_Level: int
    glucose: float
    BMI: float
    hypertension: int  
    is_smoking: int
    is_pregnant: int
    hemoglobin_a1c: float
    Diabetes_pedigree: float
    CVD_Family_History: int
    ld_value: float
    admission_tsh: float
    is_alcohol_user: int
    creatine_kinase_ck: float

# Preprocess data to match model state names
def preprocess_data(input_data, model):
    """
    Preprocesses the input data to match the state names expected by the Bayesian Network.
    Returns a tuple of (processed_evidence, original_values) dictionaries.
    """
    processed_data = {}
    original_values = {}  # Dictionary to store original values for display
    
    # Mapping of input fields to model nodes
    field_mapping = {
        'Blood_Pressure': 'BloodPressure',
        'Age': 'age',
        'Exercise_Hours_Per_Week': 'Exercise Hours Per Week',
        'Diet': 'Diet',
        'Sleep_Hours_Per_Day': 'Sleep Hours Per Day',
        'Stress_Level': 'Stress Level',
        'glucose': 'glucose',
        'BMI': 'BMI',
        'hypertension': 'hypertension',
        'is_smoking': 'is_smoking',
        'is_pregnant': 'is_pregnant',
        'hemoglobin_a1c': 'hemoglobin_a1c',
        'Diabetes_pedigree': 'Diabetes_pedigree',
        'CVD_Family_History': 'CVD_Family_History',
        'ld_value': 'ld_value',
        'admission_tsh': 'admission_tsh',
        'is_alcohol_user': 'is_alcohol_user',
        'creatine_kinase_ck': 'creatine_kinase_ck'
    }
    
    # Helper function to find matching state
    def find_matching_state(value, state_names):
        # Handle None values
        if value is None:
            return state_names[0]  # Default to first state
            
        # Handle numeric states
        if all(isinstance(x, (int, float)) for x in state_names):
            if value in state_names:
                return value
            return min(state_names, key=lambda x: abs(x - value))
        
        # Handle interval states (both string and Interval objects)
        for state in state_names:
            # String intervals like "4.0-5.0"
            if isinstance(state, str) and '-' in state:
                try:
                    lower, upper = map(float, state.split('-'))
                    if lower <= float(value) < upper:
                        return state
                except:
                    continue
            # Interval objects (pandas Interval)
            elif hasattr(state, 'right'):
                if state.left <= float(value) < state.right:
                    return state
            # Direct match for categorical values
            elif state == value:
                return state

        return state_names[-1]  # Default to last state if no match found

    for input_field, model_node in field_mapping.items():
        if input_field in input_data:
            value = input_data[input_field]
            state_names = model.get_cpds(model_node).state_names[model_node]
            
            # Store original value
            original_values[model_node] = value

            try:
                # Convert value to correct type based on expected field
                if model_node in ['hypertension', 'is_smoking', 'is_pregnant', 
                                'Diabetes_pedigree', 'CVD_Family_History', 
                                'is_alcohol_user']:
                    processed_value = int(value) if value is not None else None
                elif model_node in ['age', 'Sleep Hours Per Day', 'Stress Level']:
                    processed_value = int(value) if value is not None else None
                elif isinstance(value, (int, float)):
                    processed_value = float(value) if value is not None else None
                else:
                    # Leave as string (e.g., Diet = "Unhealthy")
                    processed_value = value

                # Find matching state
                processed_data[model_node] = find_matching_state(processed_value, state_names)

            except Exception as e:
                print(f"Error processing field '{input_field}': {e}")
                # Assign default state if processing fails
                processed_data[model_node] = state_names[0]
    
    return processed_data, original_values

# Probability calculation using pgmpy
def calculate_probabilities(model, evidence: dict):
    infer = VariableElimination(model)

    # Inference
    diabetes_result = infer.query(variables=["diabetes"], evidence=evidence, show_progress=False)
    heart_result = infer.query(variables=["heart_disease"], evidence=evidence, show_progress=False)

    diabetes_prob = float(diabetes_result.values[1])  # Assuming 1 means "Yes"
    heart_disease_prob = float(heart_result.values[1])
    
    return diabetes_prob, heart_disease_prob

# Prediction endpoint
@app.post("/predict")
async def predict_risk(data: HealthInput):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded. Please check server logs."})
    
    # Convert input to dictionary
    input_data = data.dict()

    try:
        # Preprocess input using the new function
        evidence, original_values = preprocess_data(input_data, model)

        # Calculate probabilities
        diabetes_prob, heart_disease_prob = calculate_probabilities(model, evidence)

        return {
            "Input Values": original_values,
            "Health Risk Probabilities": {
                "Diabetes": f"{diabetes_prob:.2%}",
                "Heart Disease": f"{heart_disease_prob:.2%}"
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

# Add API documentation
@app.get("/docs")
async def get_docs():
    return {"API Documentation": "Available at /docs"}

# Add API health check
@app.get("/health")
async def get_health():
    return {"status": "ok"}

# Add a root endpoint to avoid 404 on GET /
@app.get("/")
async def root():
    return {"message": "Welcome to the Bayesian Health Risk Predictor API. See /docs for API documentation."}

# Add error handling for 404 Not Found
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=404, content={"error": "Not Found"})

# Add error handling for 500 Internal Server Error
@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")