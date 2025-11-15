"""
A/B Testing logic for model comparison
"""
import random
from typing import Dict, Literal
from datetime import datetime
import json
from pathlib import Path

class ABTester:
    def __init__(self, traffic_split: Dict[str, float] = None):
        """
        Initialize A/B tester
        
        Args:
            traffic_split: Dictionary mapping model versions to traffic percentage
                          e.g., {"v1": 0.5, "v2": 0.5}
        """
        self.traffic_split = traffic_split or {"v1": 0.5, "v2": 0.5}
        self.request_log = []
        
        # Validate traffic split
        total = sum(self.traffic_split.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
    
    def select_model(self, user_id: str = None) -> str:
        """
        Select which model to use based on A/B testing strategy
        
        Args:
            user_id: Optional user ID for consistent routing
            
        Returns:
            Model version to use (e.g., "v1" or "v2")
        """
        # For now, use random selection
        # TODO: Implement user-based consistent hashing for user_id
        
        rand_value = random.random()
        cumulative = 0
        
        for model_version, split in self.traffic_split.items():
            cumulative += split
            if rand_value <= cumulative:
                return model_version
        
        # Fallback to first model
        return list(self.traffic_split.keys())[0]
    
    def log_request(self, 
                   model_version: str, 
                   prediction: Dict,
                   latency_ms: float,
                   request_id: str):
        """Log a prediction request for analysis"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "prediction": prediction["prediction"],
            "churn_probability": prediction["churn_probability"],
            "latency_ms": latency_ms
        }
        self.request_log.append(log_entry)
    
    def get_statistics(self) -> Dict:
        """Get A/B testing statistics"""
        if not self.request_log:
            return {
                "total_requests": 0,
                "models": {}
            }
        
        stats = {
            "total_requests": len(self.request_log),
            "models": {}
        }
        
        # Calculate per-model statistics
        for model_version in self.traffic_split.keys():
            model_requests = [r for r in self.request_log if r["model_version"] == model_version]
            
            if model_requests:
                latencies = [r["latency_ms"] for r in model_requests]
                predictions = [r["prediction"] for r in model_requests]
                
                stats["models"][model_version] = {
                    "request_count": len(model_requests),
                    "traffic_percentage": len(model_requests) / len(self.request_log) * 100,
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "churn_rate": sum(predictions) / len(predictions) * 100,
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies)
                }
        
        return stats
    
    def reset_logs(self):
        """Reset request logs"""
        self.request_log = []
