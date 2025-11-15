"""
Train customer churn prediction models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from pathlib import Path
import json
from datetime import datetime

class ChurnModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        """Preprocess the churn dataset"""
        # Create a copy
        df = df.copy()
        
        # Drop customerID
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Handle TotalCharges (convert to numeric, fill missing)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Convert Churn to binary
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        return df
    
    def train_model_v1(self, X_train, y_train):
        """Train Model V1: Random Forest (simpler, faster)"""
        print("\n🔧 Training Model V1: Random Forest")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("Model V1 trained")
        return model
    
    def train_model_v2(self, X_train, y_train):
        """Train Model V2: Gradient Boosting (more complex, potentially better)"""
        print("\n🔧 Training Model V2: Gradient Boosting")
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("Model V2 trained")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{model_name} Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def save_model(self, model, scaler, model_version, metrics):
        """Save model, scaler, and metadata"""
        Path("models").mkdir(exist_ok=True)
        
        # Save model
        model_path = f"models/model_{model_version}.pkl"
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = f"models/scaler_{model_version}.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Save label encoders
        encoders_path = f"models/label_encoders_{model_version}.pkl"
        joblib.dump(self.label_encoders, encoders_path)
        
        # Save metrics
        metrics_path = f"models/metrics_{model_version}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nSaved {model_version}:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Metrics: {metrics_path}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Customer Churn Prediction - Model Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("data/raw/telco_churn.csv")
    print(f"   Loaded {len(df)} records")
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Preprocess
    print("\nPreprocessing data...")
    df_processed = trainer.preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n Scaling features...")
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    # Train Model V1
    model_v1 = trainer.train_model_v1(X_train_scaled, y_train)
    metrics_v1 = trainer.evaluate_model(model_v1, X_test_scaled, y_test, "Model V1")
    trainer.save_model(model_v1, trainer.scaler, "v1", metrics_v1)
    
    # Train Model V2
    model_v2 = trainer.train_model_v2(X_train_scaled, y_train)
    metrics_v2 = trainer.evaluate_model(model_v2, X_test_scaled, y_test, "Model V2")
    trainer.save_model(model_v2, trainer.scaler, "v2", metrics_v2)
    
    # Compare models
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Model V1 (Random Forest)     - ROC AUC: {metrics_v1['roc_auc']:.4f}")
    print(f"Model V2 (Gradient Boosting) - ROC AUC: {metrics_v2['roc_auc']:.4f}")
    
    winner = "Model V2" if metrics_v2['roc_auc'] > metrics_v1['roc_auc'] else "Model V1"
    print(f"\n Best Model: {winner}")
    print("=" * 60)

if __name__ == "__main__":
    main()
