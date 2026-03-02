import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import warnings

# Suppress sklearn warnings (just in case)
warnings.filterwarnings("ignore", category=UserWarning)

def normalize_landmarks(flattened_landmarks):
    # Expects list of 63 floats
    # Wrist is the first 3 elements
    base_x, base_y, base_z = flattened_landmarks[0], flattened_landmarks[1], flattened_landmarks[2]

    # Calculate max distance from wrist
    max_dist = 0.0
    for i in range(0, 63, 3):
        x, y, z = flattened_landmarks[i], flattened_landmarks[i+1], flattened_landmarks[i+2]
        dist = ((x - base_x)**2 + (y - base_y)**2 + (z - base_z)**2)**0.5
        if dist > max_dist:
            max_dist = dist
            
    if max_dist == 0:
        max_dist = 1.0

    # Normalize
    features = []
    for i in range(0, 63, 3):
        norm_x = (flattened_landmarks[i] - base_x) / max_dist
        norm_y = (flattened_landmarks[i+1] - base_y) / max_dist
        norm_z = (flattened_landmarks[i+2] - base_z) / max_dist
        features.extend([norm_x, norm_y, norm_z])
        
    return features

# Initialize FastAPI application
app = FastAPI(title="Sign Language Translator API")

# Allow CORS so our React frontend on localhost:5173 can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, perfect for local dev
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc)
    allow_headers=["*"],  # Allows all headers
)

# Define the expected JSON payload format using Pydantic
class PredictRequest(BaseModel):
    # Expecting exactly 63 floats (21 MediaPipe hand landmarks * 3 coordinates: x, y, z)
    landmarks: List[float]

# 1. Load the trained model into memory
MODEL_PATH = "model/sign_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. Expose the POST endpoint: /predict
@app.post("/predict")
def predict_sign(request: PredictRequest):
    # Graceful error handling if the model failed to load
    if model is None:
        raise HTTPException(status_code=500, detail="The machine learning model is not loaded.")
        
    # Graceful error handling if no landmarks are provided
    if not request.landmarks:
        raise HTTPException(status_code=400, detail="No landmarks provided.")
        
    # Validation: Ensure we get exactly 63 coordinates
    if len(request.landmarks) != 63:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 63 coordinates (21 hand landmarks * 3), but got {len(request.landmarks)}."
        )

    try:
        # Normalize the incoming landmarks
        normalized_landmarks = normalize_landmarks(request.landmarks)
        
        # Convert the incoming list of floats to a 2D NumPy array
        # Scikit-learn expects shape (n_samples, n_features), so we reshape to (1, 63)
        landmarks_array = np.array(normalized_landmarks).reshape(1, -1)
        
        # To avoid scikit-learn warnings about feature names missing, 
        # we rebuild the feature names string that we trained the dataset on
        feature_names = [f"feature_{i}" for i in range(63)]
        
        # Create a pandas DataFrame for the model
        input_data = pd.DataFrame(landmarks_array, columns=feature_names)
        
        # 3. Run model prediction
        # model.predict() returns an array (e.g., ["A"]), we grab the first item [0]
        prediction = model.predict(input_data)
        predicted_label = str(prediction[0])
        
        # Calculate confidence probability safely
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                # predict_proba returns nested array of all classes' probabilities
                probabilities = model.predict_proba(input_data)
                # The highest probability belongs to the predicted class
                confidence = float(np.max(probabilities[0]))
            except Exception as e:
                print(f"Warning: Could not calculate confidence: {e}")
        
        # 4. Return predicted label & confidence as JSON
        response = {"prediction": predicted_label}
        if confidence is not None:
            response["confidence"] = confidence
            
        return response

    except Exception as e:
        # Catch any unforeseen prediction errors
        raise HTTPException(status_code=500, detail=f"Error running prediction: {str(e)}")

# Optional: Add a simple GET route so we can test the server in a web browser
@app.get("/")
def home():
    return {"message": "Sign Language API is running. Send a POST request with landmarks to /predict."}

if __name__ == "__main__":
    # Run the server using uvicorn on port 8000
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
