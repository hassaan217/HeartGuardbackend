from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from .heart_model import predict

app = FastAPI(title="Heart Disease Prediction API", 
              description="API for predicting heart disease risk",
              version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://heart-guard-ai-iota.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Define input schema - EXACTLY matching frontend field names
class HeartInput(BaseModel):
    Age: float
    Sex: int  # 0 for female, 1 for male
    chest_pain_type: int  # 1-4
    BP: float  # Blood Pressure
    Cholesterol: float
    fbs_over_120: int  # 0 for no, 1 for yes
    ekg_results: int  # 0-2
    Max_HR: float  # Max Heart Rate
    exercise_angina: int  # 0 for no, 1 for yes
    ST_depression: float
    slope_st: int  # 1-3
    num_vessels_fluro: int  # 0-3
    Thallium: float  # 3, 6, or 7

@app.get("/")
def read_root():
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This documentation",
            "POST /predict": "Make a prediction",
            "GET /health": "Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "heart-disease-prediction"}

@app.post("/predict")
def get_prediction(data: HeartInput):
    try:
        print(f"\n=== API REQUEST RECEIVED ===")
        
        # Convert to dictionary with EXACT keys that the model expects
        input_data = {
            'Age': float(data.Age),
            'Sex': int(data.Sex),
            'Chest pain type': int(data.chest_pain_type),
            'BP': float(data.BP),
            'Cholesterol': float(data.Cholesterol),
            'FBS over 120': int(data.fbs_over_120),
            'EKG results': int(data.ekg_results),
            'Max HR': float(data.Max_HR),
            'Exercise angina': int(data.exercise_angina),
            'ST depression': float(data.ST_depression),
            'Slope of ST': int(data.slope_st),
            'Number of vessels fluro': int(data.num_vessels_fluro),
            'Thallium': float(data.Thallium)  # This is numeric, not categorical
        }
        
        print(f"Processed input for model:")
        for key, value in input_data.items():
            print(f"  {key}: {value}")
        
        # Call the prediction function
        result = predict(input_data)
        
        print(f"Prediction result: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    
    
if __name__ == "__main__":
    print("🚀 Starting Heart Disease Prediction API...")
    print("📊 Make sure to train the model first using train_model.py")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
