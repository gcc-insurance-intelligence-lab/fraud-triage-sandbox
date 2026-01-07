Disclaimer
# Model Card: Fraud Triage Sandbox - Hybrid Edition

## Model Details

### Model Description

This is a **hybrid fraud detection system** for insurance claims triage, combining three complementary approaches:

1. **AI-Powered Analysis**: OpenAI GPT-4o for intelligent, contextual risk assessment
2. **ML Model Integration**: Synthetic fraud-signal-classifier-v1 for probability-based predictions
3. **Rule-Based Logic**: Transparent business rules for explainable fraud indicators

- **Developed by:** Qoder for Vercept
- **System type:** Hybrid (AI + ML + Rule-based)
- **AI Model:** OpenAI GPT-4o
- **ML Model:** fraud-signal-classifier-v1 (Random Forest)
- **ML Source:** `gcc-insurance-intelligence-lab/fraud-signal-classifier-v1`
- **Language:** Python
- **License:** MIT

### Model Sources

- **Repository:** Hugging Face Spaces
- **Demo:** Interactive Gradio interface

## Uses

### Direct Use

This tool is designed for:

- **Educational purposes**: Understanding fraud detection logic
- **Demonstration**: Showcasing rule-based triage systems
- **Prototyping**: Testing fraud detection workflows
- **Training**: Teaching insurance fraud concepts

### Downstream Use

Not applicable - this is a standalone demonstration tool.

### Out-of-Scope Use

⚠️ **This tool should NOT be used for:**

- Actual fraud detection in production environments
- Making real claims decisions
- Processing real customer data
- Regulatory compliance purposes
- Any decision that affects real people or policies

## Bias, Risks, and Limitations

### Known Limitations

1. **Synthetic ML Training Data**: The ML model is trained on 100% synthetic fraud cases
2. **Educational Purpose Only**: Not validated against real-world fraud patterns
3. **Limited Context**: Cannot consider full claim history, network analysis, or external data sources
4. **API Dependency**: AI analysis requires OpenAI API key (falls back to ML + rules if unavailable)
5. **Model Availability**: ML model requires connection to Hugging Face Hub (falls back to AI + rules if unavailable)
6. **No Continuous Learning**: Models do not adapt or retrain based on new data

### Potential Biases

- **Age Bias**: Flags younger claimants as higher risk
- **Location Bias**: Assumes certain locations have higher fraud rates
- **Claim Type Bias**: Treats certain claim types as inherently riskier

### Recommendations

Users should:

- Understand this is a **demonstration only**
- Never use for real fraud detection
- Recognize the limitations of rule-based systems
- Consider fairness implications of any fraud detection approach
- Consult with fraud experts and legal counsel for real implementations

## How to Get Started with the Model

```python
import gradio as gr

# Launch the Gradio interface
demo = gr.Interface(...)
demo.launch()
```

Or visit the Hugging Face Space to use the interactive demo.

## Training Details

### ML Model Training

The integrated ML model (`fraud-signal-classifier-v1`) was trained on synthetic fraud data:

- **Training Data**: 100% synthetic fraud cases from `fraud_cases_synthetic.csv` (235 samples)
- **Algorithm**: Random Forest Classifier (100 trees, depth=10)
- **Features**: policy_type, claimant_profile_risk, incident_pattern, document_consistency_score, anomaly_score
- **Target**: Multi-class fraud classification (Clean, Under Review, Flagged, Confirmed Fraud)
- **Encoding**: Label encoding for categorical features
- **Output**: Probability scores (0-1) mapped to Low/Medium/High buckets
- **Accuracy**: 100% on test set (small synthetic dataset)

### AI Model

OpenAI GPT-4o is used as-is via API - no custom training or fine-tuning.

### Rule-Based System

No training required - uses predefined business logic.

## Evaluation

### Testing Data, Factors & Metrics

No formal evaluation has been conducted. The system applies predefined rules without statistical validation.

### Results

Not applicable.

## Environmental Impact

Minimal - this is a lightweight rule-based system with no training requirements.

## Technical Specifications

### Model Architecture and Objective

**Architecture**: Hybrid Multi-System

**Hybrid Logic Flow**:
1. Accept claim inputs (type, sector, evidence %, behavior pattern, history count, description)
2. **ML Model Path**:
   - Map UI inputs to model features (policy_type, claimant_profile_risk, incident_pattern)
   - Calculate document_consistency_score and anomaly_score from inputs
   - Encode categorical features using label encoders
   - Generate 5-feature vector
   - Predict fraud probabilities using Random Forest (4 classes)
   - Calculate weighted fraud score from class probabilities
   - Map score to bucket (Low < 0.3, Medium 0.3-0.6, High > 0.6)
3. **AI Analysis Path**:
   - Send structured prompt to GPT-4o
   - Receive JSON analysis with risk factors, recommendations, and explanations
4. **Rule-Based Path**:
   - Apply threshold rules for evidence, behavior, claim history
   - Calculate anomaly score
   - Determine rule-based bucket
5. **Hybrid Decision**:
   - Compare ML bucket vs Rule bucket
   - If disagreement: escalate to higher severity level
   - If agreement: use consensus bucket
   - Combine all outputs with uncertainty indicators
6. **Fallback Logic**:
   - If ML fails: use AI + Rules
   - If AI fails: use ML + Rules
   - If both fail: use Rules only
7. Generate comprehensive explanation with all system outputs

### Compute Infrastructure

**Requirements**: Minimal - runs on CPU

**Dependencies**:
- Python 3.9+
- Gradio 4.44.0
- Pandas 2.1.4
- NumPy 1.26.2
- OpenAI 1.54.0
- Hugging Face Hub 0.19.0+
- Joblib 1.3.0+
- Scikit-learn 1.3.0+

## Model Card Contact

For questions or feedback, contact Vercept.

## Glossary

- **Triage**: The process of categorizing claims by priority/risk
- **Fraud Indicator**: A characteristic that suggests potential fraud
- **Risk Score**: Numerical representation of fraud likelihood
- **Rule-Based System**: Decision logic based on predefined rules rather than learned patterns

## Model Card Authors

Qoder (Vercept)

## Disclaimer

⚠️ **CRITICAL NOTICE**:

This project models generic insurance concepts common in GCC markets. All datasets are synthetic and made-up for demonstration and research purposes. No proprietary pricing, underwriting rules, policy wording, or confidential logic was used. Outputs are illustrative only and require human review. Not to be used for any pricing, reserving, claim approval, or policy issuance.

## Human-In-The-Loop

No AI component here issues approvals, denials, or financial outcomes. All outputs require human verification and decision-making.

---

This is a **synthetic demonstration tool only**. All data, scenarios, and outputs are fictional. This tool is not suitable for production use, real fraud detection, or any decision-making that affects actual insurance claims or customers. All outputs are advisory only and should not be relied upon for any real-world purpose.
