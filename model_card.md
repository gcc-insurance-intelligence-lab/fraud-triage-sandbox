Disclaimer
# Model Card: Fraud Triage Sandbox

## Model Details

### Model Description

This is a **rule-based fraud detection system** for insurance claims triage. It does not use machine learning models but instead applies configurable business rules to assess fraud risk.

- **Developed by:** Qoder for Vercept
- **Model type:** Rule-based decision system
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

1. **Simplified Rules**: Real fraud detection requires much more sophisticated analysis
2. **No Machine Learning**: Does not learn from data or adapt to new patterns
3. **Limited Context**: Cannot consider claim history, network analysis, or external data
4. **Synthetic Data Only**: Designed for demonstration with fake data
5. **No Validation**: Rules are illustrative and not validated against real fraud patterns

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

Not applicable - this is a rule-based system, not a trained model.

## Evaluation

### Testing Data, Factors & Metrics

No formal evaluation has been conducted. The system applies predefined rules without statistical validation.

### Results

Not applicable.

## Environmental Impact

Minimal - this is a lightweight rule-based system with no training requirements.

## Technical Specifications

### Model Architecture and Objective

**Architecture**: Rule-based decision tree

**Logic Flow**:
1. Accept claim inputs (amount, age, type, location, filing day)
2. Apply configurable threshold rules
3. Count fraud indicators
4. Calculate risk score
5. Assign triage category
6. Generate explanation

### Compute Infrastructure

**Requirements**: Minimal - runs on CPU

**Dependencies**:
- Python 3.9+
- Gradio 4.44.0
- Pandas 2.1.4
- NumPy 1.26.2

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
