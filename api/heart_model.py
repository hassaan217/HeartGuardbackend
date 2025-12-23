import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Get directory of this file (api/)
BASE_DIR = os.path.dirname(__file__)

# Load model with relative path


# Load trained model, scaler, and feature columns
def load_model():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
        return model, scaler, feature_columns
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        raise


# Preprocess input data - FIXED with EXACT feature order
def preprocess_input(input_data, feature_columns, scaler):
    """Convert input data to DataFrame with EXACT same features and order as training"""
    
    # Extract values
    chest_pain_type = int(input_data.get('Chest pain type', 1))
    ekg_results = int(input_data.get('EKG results', 0))
    slope_st = int(input_data.get('Slope of ST', 1))
    num_vessels = int(input_data.get('Number of vessels fluro', 0))
    
    # Create dictionary with features in EXACT training order
    processed_data = {}
    
    # Add features in EXACT order from training:
    # Order from your training output:
    # 0: Age, 1: Sex, 2: BP, 3: Cholesterol, 4: FBS over 120,
    # 5: EKG results_1, 6: EKG results_2, 7: Max HR, 8: Exercise angina,
    # 9: ST depression, 10: Slope of ST_2, 11: Slope of ST_3,
    # 12: Number of vessels fluro_1, 13: Number of vessels fluro_2,
    # 14: Number of vessels fluro_3, 15: Thallium,
    # 16: Chest pain type_2, 17: Chest pain type_3, 18: Chest pain type_4
    
    # 0: Age
    processed_data['Age'] = float(input_data['Age'])
    
    # 1: Sex
    processed_data['Sex'] = int(input_data['Sex'])
    
    # 2: BP
    processed_data['BP'] = float(input_data['BP'])
    
    # 3: Cholesterol
    processed_data['Cholesterol'] = float(input_data['Cholesterol'])
    
    # 4: FBS over 120
    processed_data['FBS over 120'] = int(input_data['FBS over 120'])
    
    # 5: EKG results_1
    processed_data['EKG results_1'] = 1 if ekg_results == 1 else 0
    
    # 6: EKG results_2
    processed_data['EKG results_2'] = 1 if ekg_results == 2 else 0
    
    # 7: Max HR
    processed_data['Max HR'] = float(input_data['Max HR'])
    
    # 8: Exercise angina
    processed_data['Exercise angina'] = int(input_data['Exercise angina'])
    
    # 9: ST depression
    processed_data['ST depression'] = float(input_data['ST depression'])
    
    # 10: Slope of ST_2
    processed_data['Slope of ST_2'] = 1 if slope_st == 2 else 0
    
    # 11: Slope of ST_3
    processed_data['Slope of ST_3'] = 1 if slope_st == 3 else 0
    
    # 12: Number of vessels fluro_1
    processed_data['Number of vessels fluro_1'] = 1 if num_vessels == 1 else 0
    
    # 13: Number of vessels fluro_2
    processed_data['Number of vessels fluro_2'] = 1 if num_vessels == 2 else 0
    
    # 14: Number of vessels fluro_3
    processed_data['Number of vessels fluro_3'] = 1 if num_vessels == 3 else 0
    
    # 15: Thallium
    processed_data['Thallium'] = float(input_data['Thallium'])
    
    # 16: Chest pain type_2
    processed_data['Chest pain type_2'] = 1 if chest_pain_type == 2 else 0
    
    # 17: Chest pain type_3
    processed_data['Chest pain type_3'] = 1 if chest_pain_type == 3 else 0
    
    # 18: Chest pain type_4
    processed_data['Chest pain type_4'] = 1 if chest_pain_type == 4 else 0
    
    # Create DataFrame with EXACT column order from training
    df = pd.DataFrame(columns=feature_columns)
    
    # Fill in values for each column in the exact order
    for col in feature_columns:
        if col in processed_data:
            df[col] = [processed_data[col]]
        else:
            df[col] = [0]  # Default value for any missing columns
    
    # Scale numeric columns - in the SAME order as during training
    # Note: Thallium is NOT in the numeric columns list from your training
    num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    
    # Create a copy and scale only numeric columns
    df_scaled = df.copy()
    
    # Print for debugging
    print(f"\nDataFrame before scaling (first 5 columns):")
    for i, col in enumerate(df.columns[:5]):
        print(f"  {i}: {col} = {df[col].values[0]}")
    
    # Scale the numeric columns
    if scaler:
        df_scaled[num_cols] = scaler.transform(df[num_cols])
        
        print(f"\nDataFrame after scaling (first 5 columns):")
        for i, col in enumerate(df_scaled.columns[:5]):
            print(f"  {i}: {col} = {df_scaled[col].values[0]}")
    
    return df_scaled

# Make prediction
def predict(input_data: dict):
    """Takes JSON-like input from FastAPI, preprocesses, and predicts."""
    
    # Load model, scaler, and feature columns
    model, scaler, feature_columns = load_model()
    if model is None or scaler is None or feature_columns is None:
        return {"error": "Model not loaded properly"}
    
    print(f"\n=== PREDICTION REQUEST ===")
    print(f"Input data received:")
    for key, value in input_data.items():
        print(f"  {key}: {value} (type: {type(value).__name__})")
    
    # Preprocess input with exact feature order
    try:
        df = preprocess_input(input_data, feature_columns, scaler)
        
        # Verify the column order
        print(f"\nFinal DataFrame columns ({len(df.columns)} total):")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col} = {df[col].values[0]:.4f}")
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        print(f"\nPrediction: {prediction}")
        print(f"Probability of heart disease: {probability:.4f}")
        
        # Convert prediction to readable format
        result = "Presence" if prediction == 1 else "Absence"
        confidence = f"{probability:.2%}"
        
        return {
            "prediction": result,
            "probability": float(probability),
            "confidence": confidence,
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        }
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}
