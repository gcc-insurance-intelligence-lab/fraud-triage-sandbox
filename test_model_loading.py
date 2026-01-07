#!/usr/bin/env python3
"""
Test script to verify ML model loading from Hugging Face Hub
"""

import sys
import json
from huggingface_hub import hf_hub_download
import joblib

MODEL_REPO = "gcc-insurance-intelligence-lab/fraud-signal-classifier-v1"

print("Testing ML Model Loading...")
print("=" * 50)

try:
    print(f"\n1. Downloading model artifacts from {MODEL_REPO}...")
    
    # Download model artifacts
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl")
    print(f"   ✓ model.pkl downloaded to: {model_path}")
    
    encoders_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoders.pkl")
    print(f"   ✓ label_encoders.pkl downloaded to: {encoders_path}")
    
    features_path = hf_hub_download(repo_id=MODEL_REPO, filename="feature_names.json")
    print(f"   ✓ feature_names.json downloaded to: {features_path}")
    
    print("\n2. Loading model artifacts...")
    
    # Load model
    ml_model = joblib.load(model_path)
    print(f"   ✓ Model loaded: {type(ml_model).__name__}")
    
    # Load encoders
    label_encoders = joblib.load(encoders_path)
    print(f"   ✓ Label encoders loaded: {list(label_encoders.keys())}")
    
    # Load feature names
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    print(f"   ✓ Feature names loaded: {feature_names}")
    
    print("\n3. Testing model prediction...")
    
    # Test prediction with sample data
    import numpy as np
    
    # Sample encoded features (using first category for each encoder)
    claim_type_enc = 0
    sector_enc = 0
    evidence_pct = 60
    behavior_enc = 0
    claim_history_count = 1
    
    features = [claim_type_enc, sector_enc, evidence_pct, behavior_enc, claim_history_count]
    X = np.array(features).reshape(1, -1)
    
    # Get prediction
    proba = ml_model.predict_proba(X)[0]
    prediction = ml_model.predict(X)[0]
    
    print(f"   ✓ Prediction successful!")
    print(f"   - Predicted class: {prediction}")
    print(f"   - Probabilities: Non-Fraud={proba[0]:.3f}, Fraud={proba[1]:.3f}")
    
    # Map to bucket
    fraud_proba = proba[1]
    if fraud_proba < 0.3:
        bucket = "Low"
    elif fraud_proba < 0.7:
        bucket = "Medium"
    else:
        bucket = "High"
    
    print(f"   - Risk Bucket: {bucket}")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("Model loading and inference working correctly.")
    print("=" * 50)
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 50)
    print("❌ TEST FAILED")
    print("=" * 50)
    sys.exit(1)
