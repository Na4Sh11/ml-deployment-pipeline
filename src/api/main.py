"""
FastAPI application for churn prediction with A/B testing
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional
import time
import uuid
from contextlib import asynccontextmanager

from src.models.predict import ChurnPredictor
from src.api.ab_testing import ABTester
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Prometheus metrics
prediction_counter = Counter(
    'predictions_total', 
    'Total number of predictions',
    ['model_version']
)
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version']
)
churn_predictions = Counter(
    'churn_predictions_total',
    'Total churn predictions',
    ['model_version', 'prediction']
)

# Global variables for models and A/B tester
models = {}
ab_tester = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global models, ab_tester
    
    print(" Loading models...")
    try:
        models["v1"] = ChurnPredictor("v1")
        models["v2"] = ChurnPredictor("v2")
        print("Models loaded successfully")
    except Exception as e:
        print(f" Failed to load models: {e}")
        raise
    
    # Initialize A/B tester
    ab_tester = ABTester(traffic_split={"v1": 0.5, "v2": 0.5})
    print(" A/B tester initialized")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML deployment with A/B testing and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response
class CustomerData(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=840.50)

class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    prediction: int
    churn_probability: float
    will_churn: bool
    confidence: float
    latency_ms: float

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "models": "/models",
            "ab_stats": "/ab-test/stats",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }

@app.get("/models")
async def get_models_info():
    """Get information about all loaded models"""
    models_info = {}
    for version, predictor in models.items():
        models_info[version] = predictor.get_model_info()
    return models_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData, user_id: Optional[str] = None):
    """
    Predict customer churn using A/B testing
    
    The API will automatically route traffic between model versions based on A/B testing configuration.
    """
    try:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Select model using A/B testing
        selected_model = ab_tester.select_model(user_id)
        
        # Make prediction
        start_time = time.time()
        prediction = models[selected_model].predict(customer.dict())
        latency_ms = (time.time() - start_time) * 1000
        
        # Update Prometheus metrics
        prediction_counter.labels(model_version=selected_model).inc()
        prediction_latency.labels(model_version=selected_model).observe(latency_ms / 1000)
        churn_predictions.labels(
            model_version=selected_model,
            prediction=str(prediction["prediction"])
        ).inc()
        
        # Log request for A/B testing analysis
        ab_tester.log_request(selected_model, prediction, latency_ms, request_id)
        
        # Prepare response
        response = PredictionResponse(
            request_id=request_id,
            model_version=selected_model,
            prediction=prediction["prediction"],
            churn_probability=prediction["churn_probability"],
            will_churn=prediction["will_churn"],
            confidence=prediction["confidence"],
            latency_ms=latency_ms
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/ab-test/stats")
async def get_ab_stats():
    """Get A/B testing statistics"""
    return ab_tester.get_statistics()

@app.post("/ab-test/reset")
async def reset_ab_stats():
    """Reset A/B testing logs"""
    ab_tester.reset_logs()
    return {"message": "A/B testing logs reset successfully"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
