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

# Fraud Triage Sandbox

## Overview

An interactive demonstration of rule-based fraud detection for insurance claims. This sandbox allows users to input claim details and receive a fraud risk assessment based on configurable detection rules.

---

## Disclaimer

This project models generic insurance concepts common in GCC markets. All datasets are synthetic and made-up for demonstration and research purposes. No proprietary pricing, underwriting rules, policy wording, or confidential logic was used. Outputs are illustrative only and require human review. Not to be used for any pricing, reserving, claim approval, or policy issuance.

## Human-In-The-Loop

No AI component here issues approvals, denials, or financial outcomes. All outputs require human verification and decision-making.

---

## Features

- **Interactive Claim Input**: Enter claim type, sector, evidence percentage, behavior pattern, and claim history
- **Rule-Based Detection**: Configurable business rules for fraud indicators
- **Anomaly Scoring**: Automatic calculation of fraud anomaly score (0-1)
- **Fraud Likelihood Buckets**: Classification into Low/Medium/High risk categories
- **Uncertainty Scoring**: Confidence assessment based on available information
- **Human Review Warnings**: Mandatory human-in-the-loop enforcement
- **Detailed Explanations**: Transparent reasoning for each risk assessment

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
- **Dependencies**: pandas, numpy
- **Model**: Rule-based logic (no ML model)

## License

MIT License

---

**Built by Qoder for Vercept**
