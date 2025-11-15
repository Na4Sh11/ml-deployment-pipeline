"""
Download and prepare the Telco Customer Churn dataset
"""
import pandas as pd
import os
from pathlib import Path

def download_data():
    """Download the dataset from a reliable source"""
    
    # Create data directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # URL for the dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    print("📥 Downloading Telco Customer Churn dataset...")
    df = pd.read_csv(url)
    
    # Save raw data
    df.to_csv("data/raw/telco_churn.csv", index=False)
    print(f"✅ Dataset downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Saved to: data/raw/telco_churn.csv")
    
    # Display basic info
    print("\nDataset Info:")
    print(f"   Total Customers: {len(df)}")
    print(f"   Churned Customers: {df['Churn'].value_counts()['Yes']} ({df['Churn'].value_counts()['Yes']/len(df)*100:.1f}%)")
    print(f"   Retained Customers: {df['Churn'].value_counts()['No']} ({df['Churn'].value_counts()['No']/len(df)*100:.1f}%)")
    
    print("\nFeatures:")
    print(df.columns.tolist())
    
    return df

if __name__ == "__main__":
    download_data()
