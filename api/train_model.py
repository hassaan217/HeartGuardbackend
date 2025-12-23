import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_and_save_model():
    # Load your dataset
    df = pd.read_csv('Heart_Disease_Prediction.csv')
    
    # Print column names for debugging
    print("Original columns:", df.columns.tolist())
    
    # Preprocess the target variable
    prediction_map = {
        "Presence": 1,
        "Absence": 0 
    }
    df["Heart Disease"] = df["Heart Disease"].map(prediction_map)
    
    # Initialize preprocessors
    scaler = StandardScaler()
    le = LabelEncoder()
    
    # Define numeric columns
    num_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
    
    # Scale numeric columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Encode binary columns
    binary_cols = [
        'Sex',
        'FBS over 120',
        'Exercise angina',
        'Heart Disease'
    ]
    
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    
    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=[
            'Chest pain type',
            'EKG results',
            'Slope of ST',
            'Number of vessels fluro'
        ],
        drop_first=True
    )
    
    # Print all columns after preprocessing
    print("\nAll columns after preprocessing:")
    print(df.columns.tolist())
    
    # Define features - IMPORTANT: This must match exactly
    feature_columns = [
        'Age', 'Sex', 'BP', 'Cholesterol', 'FBS over 120', 
        'EKG results_1', 'EKG results_2', 'Max HR', 
        'Exercise angina', 'ST depression', 'Slope of ST_2', 
        'Slope of ST_3', 'Number of vessels fluro_1', 
        'Number of vessels fluro_2', 'Number of vessels fluro_3', 
        'Thallium', 'Chest pain type_2', 'Chest pain type_3', 
        'Chest pain type_4'
    ]
    
    # Check if all features exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"\n❌ Missing columns: {missing_cols}")
        # Add missing columns with 0 values
        for col in missing_cols:
            df[col] = 0
    
    # Reorder columns to match feature_columns order
    X = df[feature_columns]
    y = df['Heart Disease']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression model
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Feature columns ({len(feature_columns)}):")
    for i, col in enumerate(feature_columns):
        print(f"  {i}: {col}")
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    joblib.dump(model, 'log_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save the feature columns for consistency
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    print("\n✅ Model and scaler saved successfully!")
    print("Files created:")
    print("- log_model.pkl")
    print("- scaler.pkl")
    print("- feature_columns.pkl")
    
    # Create a sample input for testing
    sample_input = {
        'Age': 54.0,
        'Sex': 1,
        'Chest pain type': 3,
        'BP': 130.0,
        'Cholesterol': 240.0,
        'FBS over 120': 0,
        'EKG results': 0,
        'Max HR': 140.0,
        'Exercise angina': 0,
        'ST depression': 2.5,
        'Slope of ST': 2,
        'Number of vessels fluro': 0,
        'Thallium': 3.0
    }
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    train_and_save_model()