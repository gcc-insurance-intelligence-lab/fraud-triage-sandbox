import gradio as gr
import random

# Fraud detection logic (rule-based for demo)
def calculate_fraud_likelihood(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count):
    """Calculate fraud likelihood based on rule-based logic."""
    
    # Initialize anomaly score
    anomaly_score = 0.0
    
    # Evidence percentage factor (lower evidence = higher risk)
    if evidence_pct < 30:
        anomaly_score += 0.35
    elif evidence_pct < 50:
        anomaly_score += 0.20
    elif evidence_pct < 70:
        anomaly_score += 0.10
    
    # Behavior pattern factor
    if behavior_pattern == "Inconsistent statements":
        anomaly_score += 0.25
    elif behavior_pattern == "Rushed filing":
        anomaly_score += 0.20
    elif behavior_pattern == "Multiple similar claims":
        anomaly_score += 0.30
    elif behavior_pattern == "Suspicious timing":
        anomaly_score += 0.15
    
    # Claim history factor
    if claim_history_count >= 5:
        anomaly_score += 0.25
    elif claim_history_count >= 3:
        anomaly_score += 0.15
    elif claim_history_count >= 2:
        anomaly_score += 0.08
    
    # Claim type factor
    high_risk_types = ["Theft", "Total Loss", "Fire"]
    if claim_type in high_risk_types:
        anomaly_score += 0.15
    
    # Sector factor
    if sector == "Commercial":
        anomaly_score += 0.05  # Slightly higher complexity
    
    # Cap at 1.0
    anomaly_score = min(anomaly_score, 1.0)
    
    return anomaly_score

def get_fraud_bucket(anomaly_score):
    """Classify fraud likelihood into buckets."""
    if anomaly_score >= 0.65:
        return "High"
    elif anomaly_score >= 0.35:
        return "Medium"
    else:
        return "Low"

def calculate_uncertainty(evidence_pct, claim_history_count):
    """Calculate uncertainty score based on available information."""
    uncertainty = 0.0
    
    # Low evidence increases uncertainty
    if evidence_pct < 40:
        uncertainty += 0.40
    elif evidence_pct < 60:
        uncertainty += 0.25
    elif evidence_pct < 80:
        uncertainty += 0.15
    
    # Limited claim history increases uncertainty
    if claim_history_count == 0:
        uncertainty += 0.30
    elif claim_history_count == 1:
        uncertainty += 0.15
    
    return min(uncertainty, 1.0)

def generate_explanation(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, anomaly_score, bucket):
    """Generate detailed explanation of the fraud assessment."""
    
    explanation = f"### Fraud Triage Assessment\n\n"
    explanation += f"**Claim Type:** {claim_type}\n"
    explanation += f"**Sector:** {sector}\n"
    explanation += f"**Evidence Provided:** {evidence_pct}%\n"
    explanation += f"**Behavior Pattern:** {behavior_pattern}\n"
    explanation += f"**Claim History Count:** {claim_history_count}\n\n"
    
    explanation += f"---\n\n"
    explanation += f"**Anomaly Score:** {anomaly_score:.2f} / 1.00\n"
    explanation += f"**Fraud Likelihood Bucket:** {bucket}\n\n"
    
    explanation += f"#### Risk Factors Identified:\n\n"
    
    factors = []
    
    if evidence_pct < 50:
        factors.append(f"⚠️ Low evidence documentation ({evidence_pct}%)")
    
    if behavior_pattern in ["Inconsistent statements", "Multiple similar claims", "Suspicious timing"]:
        factors.append(f"⚠️ Concerning behavior pattern: {behavior_pattern}")
    
    if claim_history_count >= 3:
        factors.append(f"⚠️ High claim frequency ({claim_history_count} prior claims)")
    
    if claim_type in ["Theft", "Total Loss", "Fire"]:
        factors.append(f"⚠️ High-risk claim type: {claim_type}")
    
    if factors:
        for factor in factors:
            explanation += f"- {factor}\n"
    else:
        explanation += "- ✅ No significant risk factors detected\n"
    
    return explanation

def generate_human_review_warning(bucket, uncertainty):
    """Generate human review warning based on bucket and uncertainty."""
    
    warning = "\n---\n\n### ⚠️ Human Review Required\n\n"
    
    if bucket == "High":
        warning += "**CRITICAL:** This claim has been flagged as HIGH RISK for potential fraud. "
        warning += "Immediate human investigation is required before any processing decisions are made.\n\n"
        warning += "**Recommended Actions:**\n"
        warning += "- Assign to fraud investigation team\n"
        warning += "- Request additional documentation\n"
        warning += "- Conduct claimant interview\n"
        warning += "- Verify all evidence independently\n"
    elif bucket == "Medium":
        warning += "**CAUTION:** This claim shows moderate fraud indicators. "
        warning += "Enhanced review and verification are required.\n\n"
        warning += "**Recommended Actions:**\n"
        warning += "- Request additional supporting documentation\n"
        warning += "- Verify claimant information\n"
        warning += "- Review claim history in detail\n"
        warning += "- Consider field investigation if warranted\n"
    else:
        warning += "**STANDARD:** This claim shows low fraud indicators, but human review is still required. "
        warning += "Follow standard claims processing procedures.\n\n"
        warning += "**Recommended Actions:**\n"
        warning += "- Standard documentation review\n"
        warning += "- Verify basic claim information\n"
        warning += "- Process according to normal workflow\n"
    
    if uncertainty > 0.5:
        warning += "\n⚠️ **HIGH UNCERTAINTY:** Limited information available. Additional data collection strongly recommended.\n"
    
    warning += "\n---\n\n"
    warning += "**IMPORTANT:** This is an advisory tool only. All fraud determinations must be made by qualified human investigators. "
    warning += "No automated system should make final decisions on fraud accusations or claim denials."
    
    return warning

def analyze_fraud_risk(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count):
    """Main function to analyze fraud risk."""
    
    # Calculate anomaly score and fraud likelihood
    anomaly_score = calculate_fraud_likelihood(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count)
    
    # Get fraud bucket
    bucket = get_fraud_bucket(anomaly_score)
    
    # Calculate uncertainty
    uncertainty = calculate_uncertainty(evidence_pct, claim_history_count)
    
    # Generate explanation
    explanation = generate_explanation(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, anomaly_score, bucket)
    
    # Generate human review warning
    warning = generate_human_review_warning(bucket, uncertainty)
    
    # Combine outputs
    full_output = explanation + warning
    
    # Create summary
    summary = f"""
**Fraud Likelihood:** {bucket}
**Anomaly Score:** {anomaly_score:.2f}
**Uncertainty Score:** {uncertainty:.2f}
**Human Review:** Required
    """
    
    return full_output, summary, anomaly_score, uncertainty

# Create Gradio interface
with gr.Blocks(title="Fraud Triage Sandbox", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Fraud Triage Sandbox
    
    **Rule-Based Fraud Detection System for Insurance Claims**
    
    This tool demonstrates a rule-based fraud triage system for insurance claims. 
    Enter claim details to receive a fraud risk assessment and triage recommendation.
    
    ## ⚠️ MANDATORY DISCLAIMER
    
    **This is a demonstration tool for educational purposes only.**
    
    - ✅ All outputs are **advisory only** and require human verification
    - ✅ No AI component issues approvals, denials, or financial outcomes
    - ✅ All fraud determinations must be made by qualified human investigators
    - ✅ This tool uses **rule-based logic only** - not machine learning
    - ✅ No real insurance company data or proprietary rules are used
    - ✅ Not for use in actual claim approval, pricing, or reserving decisions
    
    **Human-in-the-loop is mandatory for all fraud investigations.**
    """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Claim Information")
            
            claim_type = gr.Dropdown(
                choices=["Property Damage", "Theft", "Collision", "Fire", "Water Damage", "Total Loss", "Liability"],
                label="Claim Type",
                value="Property Damage"
            )
            
            sector = gr.Radio(
                choices=["Personal", "Commercial"],
                label="Sector",
                value="Personal"
            )
            
            evidence_pct = gr.Slider(
                minimum=0,
                maximum=100,
                step=5,
                label="Evidence Percentage (%)",
                value=60,
                info="Percentage of required documentation provided"
            )
            
            behavior_pattern = gr.Dropdown(
                choices=["Normal", "Inconsistent statements", "Rushed filing", "Multiple similar claims", "Suspicious timing", "Evasive responses"],
                label="Behavior Pattern",
                value="Normal"
            )
            
            claim_history_count = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                label="Claim History Count",
                value=1,
                info="Number of previous claims filed"
            )
            
            analyze_btn = gr.Button("🔍 Analyze Fraud Risk", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### Assessment Results")
            
            summary_output = gr.Textbox(
                label="Quick Summary",
                lines=5,
                interactive=False
            )
            
            anomaly_output = gr.Number(
                label="Anomaly Score (0-1)",
                interactive=False
            )
            
            uncertainty_output = gr.Number(
                label="Uncertainty Score (0-1)",
                interactive=False
            )
    
    with gr.Row():
        detailed_output = gr.Markdown(label="Detailed Analysis")
    
    analyze_btn.click(
        fn=analyze_fraud_risk,
        inputs=[claim_type, sector, evidence_pct, behavior_pattern, claim_history_count],
        outputs=[detailed_output, summary_output, anomaly_output, uncertainty_output]
    )
    
    with gr.Accordion("ℹ️ About This Tool", open=False):
        gr.Markdown("""
        ## How It Works
        
        This fraud triage system uses **configurable business rules** to assess fraud risk. It does NOT use machine learning models.
        
        ### Risk Factors Evaluated:
        
        1. **Evidence Percentage**: Lower documentation completeness increases risk
        2. **Behavior Pattern**: Suspicious behaviors (inconsistent statements, rushed filing, etc.) raise flags
        3. **Claim History**: Multiple previous claims increase scrutiny
        4. **Claim Type**: Certain claim types (theft, total loss, fire) carry higher inherent risk
        5. **Sector**: Commercial claims may have different risk profiles
        
        ### Fraud Likelihood Buckets:
        
        - **Low (0.0 - 0.34)**: Standard processing with normal verification
        - **Medium (0.35 - 0.64)**: Enhanced review and additional documentation required
        - **High (0.65 - 1.0)**: Immediate escalation to fraud investigation team
        
        ### Uncertainty Score:
        
        Indicates confidence in the assessment based on available information. Higher uncertainty suggests more data is needed.
        
        ### Educational Purposes:
        
        - **Prototyping**: Test fraud detection workflows
        - **Training**: Teach insurance fraud concepts
        - **Demonstration**: Showcase rule-based triage systems
        - **Testing**: Validate fraud detection logic
        
        ### Compliance & Safety:
        
        - ✅ No real insurer names or proprietary information
        - ✅ No actuarial formulas or pricing models
        - ✅ No KYC or sensitive personal fields
        - ✅ All outputs marked as advisory only
        - ✅ Human-in-the-loop enforced
        
        ### Limitations:
        
        - Simplified rule-based logic (real systems use ML + human expertise)
        - No integration with actual claims databases
        - No real-time fraud network analysis
        - Educational demonstration only
        
        **Built by Qoder for Vercept**
        """
        )

if __name__ == "__main__":
    demo.launch()
