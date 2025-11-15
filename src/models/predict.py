"""
Prediction logic for churn models
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

class ChurnPredictor:
    def __init__(self, model_version: str = "v1"):
        """Initialize predictor with a specific model version"""
        self.model_version = model_version
        self.model_path = f"models/model_{model_version}.pkl"
        self.scaler_path = f"models/scaler_{model_version}.pkl"
        self.encoders_path = f"models/label_encoders_{model_version}.pkl"
        self.metrics_path = f"models/metrics_{model_version}.json"
        
        # Load model artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and encoders"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoders = joblib.load(self.encoders_path)
            
            # Load metrics
            with open(self.metrics_path, 'r') as f:
                self.metrics = json.load(f)
            
            print(f"Loaded {self.model_version} successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_version}: {str(e)}")
    
    def preprocess_input(self, data: Dict) -> np.ndarray:
        """Preprocess input data for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Drop customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Handle TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def predict(self, data: Dict) -> Dict:
        """Make a prediction"""
        # Preprocess
        X = self.preprocess_input(data)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return {
            "model_version": self.model_version,
            "prediction": int(prediction),
            "churn_probability": float(probability[1]),
            "will_churn": bool(prediction == 1),
            "confidence": float(max(probability))
        }
    
    def get_model_info(self) -> Dict:
        """Get model information and metrics"""
        return {
            "model_version": self.model_version,
            "metrics": self.metrics
        }
