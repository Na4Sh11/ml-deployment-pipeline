"""
Load testing script using Locust
"""
from locust import HttpUser, task, between
import random

class ChurnPredictionUser(HttpUser):
    wait_time = between(0.5, 2)  # Wait 0.5-2 seconds between requests
    
    # Sample customer profiles
    customer_profiles = [
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 840.50
        },
        {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": 85.0
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 48,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 110.0,
            "TotalCharges": 5280.0
        },
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "Yes",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 55.0,
            "TotalCharges": 1320.0
        }
    ]
    
    @task(10)
    def predict_churn(self):
        """Make a prediction request"""
        customer = random.choice(self.customer_profiles)
        self.client.post("/predict", json=customer)
    
    @task(2)
    def check_health(self):
        """Check API health"""
        self.client.get("/health")
    
    @task(1)
    def get_ab_stats(self):
        """Get A/B testing statistics"""
        self.client.get("/ab-test/stats")
    
    @task(1)
    def get_models_info(self):
        """Get models information"""
        self.client.get("/models")
