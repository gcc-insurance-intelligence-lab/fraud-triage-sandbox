---
title: Fraud Triage Sandbox
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Fraud Triage Sandbox - Hybrid Edition

## Overview

An interactive demonstration of **hybrid fraud detection** for insurance claims, combining:
- **AI-Powered Analysis** using OpenAI GPT-4o for intelligent risk assessment
- **ML Model Integration** using fraud-signal-classifier-v1 (Random Forest trained on synthetic data)
- **Rule-Based Logic** for transparent, explainable fraud indicators

This sandbox allows users to input claim details and receive comprehensive fraud risk assessments from three detection systems working together in real-time.

---

## Disclaimer

This project models generic insurance concepts common in GCC markets. All datasets are synthetic and made-up for demonstration and research purposes. No proprietary pricing, underwriting rules, policy wording, or confidential logic was used. Outputs are illustrative only and require human review. Not to be used for any pricing, reserving, claim approval, or policy issuance.

## Human-In-The-Loop

No AI component here issues approvals, denials, or financial outcomes. All outputs require human verification and decision-making.

---

## 🔬 Hybrid Mode Enabled

This application now operates in **Hybrid Mode**, combining three complementary fraud detection approaches:

### 1. AI-Powered Analysis (OpenAI GPT-4o)
- Natural language understanding of claim descriptions
- Contextual risk factor identification
- Protective factor recognition
- Investigation recommendations
- Explainable reasoning

### 2. ML Model Predictions (fraud-signal-classifier-v1)
- Trained on synthetic fraud cases
- Probability-based fraud scoring (0-1)
- Automated bucket classification (Low/Medium/High)
- Feature-based pattern recognition
- Model source: `gcc-insurance-intelligence-lab/fraud-signal-classifier-v1`

### 3. Rule-Based Logic
- Transparent business rules
- Evidence completeness checks
- Behavior pattern analysis
- Claim history evaluation
- Sector-specific risk factors

### Hybrid Decision Logic

The system intelligently combines all three approaches:
- **Agreement**: When ML and rules agree, confidence is high
- **Disagreement**: When systems disagree, the higher severity level is selected for safety
- **Escalation**: Automatic escalation when uncertainty is detected
- **Fallback**: If AI or ML fails, rule-based system ensures continuity

## Features

- **Interactive Claim Input**: Enter claim type, sector, evidence percentage, behavior pattern, and claim history
- **Multi-System Detection**: AI + ML + Rules working together
- **Anomaly Scoring**: Automatic calculation of fraud anomaly score (0-1)
- **ML Fraud Probability**: Machine learning-based fraud likelihood (0-1)
- **Fraud Likelihood Buckets**: Classification into Low/Medium/High risk categories
- **Uncertainty Scoring**: Confidence assessment based on available information
- **Human Review Warnings**: Mandatory human-in-the-loop enforcement
- **Detailed Explanations**: Transparent reasoning for each risk assessment
- **Graceful Degradation**: Automatic fallback if any component fails

## Detection Rules

The system evaluates claims based on:

1. **Evidence Percentage**: Lower documentation completeness increases risk
2. **Behavior Pattern**: Suspicious behaviors (inconsistent statements, rushed filing, multiple similar claims)
3. **Claim History Count**: Multiple previous claims increase scrutiny
4. **Claim Type**: High-risk types (Theft, Total Loss, Fire) carry higher risk
5. **Sector**: Commercial vs Personal insurance differences

## Risk Levels

- **Low Risk (0-2 flags)**: Auto-Approve
- **Medium Risk (3-4 flags)**: Manual Review Required
- **High Risk (5+ flags)**: Flag for Investigation

## Usage

1. Enter claim details in the input fields
2. Adjust detection thresholds if desired
3. Click "Analyze Claim" to receive fraud assessment
4. Review the risk score, triage decision, and detailed explanation

## Compliance & Safety

⚠️ **IMPORTANT DISCLAIMERS**:

- All data and scenarios are **100% synthetic**
- This is a **demonstration tool only**
- Not intended for actual fraud detection or claims processing
- All outputs are **advisory only** and should not be used for real decisions
- No real insurer names, policies, or actuarial formulas are used
- No KYC fields, pricing, or quoting functionality included

## Technical Details

- **Framework**: Gradio 4.44.0
- **Language**: Python 3.9+
- **AI Model**: OpenAI GPT-4o
- **ML Model**: fraud-signal-classifier-v1 (Random Forest on synthetic fraud cases)
- **ML Source**: Hugging Face Hub - `gcc-insurance-intelligence-lab/fraud-signal-classifier-v1`
- **Dependencies**: gradio, pandas, numpy, openai, huggingface-hub, joblib, scikit-learn
- **Architecture**: Hybrid (AI + ML + Rule-based)

## Configuration

### OpenAI API Key
To enable AI-powered analysis, set the `OPENAI_API_KEY` environment variable or Hugging Face Secret.
Without the API key, the system will use ML + rule-based fallback.

### ML Model
The ML model is automatically downloaded from Hugging Face Hub on startup.
If the model is unavailable, the system falls back to AI + rule-based analysis.

## License

MIT License

---

**Built by Qoder for Vercept**
