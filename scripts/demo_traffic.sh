#!/bin/bash

echo " Generating demo traffic for screenshots..."

for i in {1..50}; do
  curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
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
    }' > /dev/null
  
  sleep 0.2
done

echo "Traffic generation complete!"
