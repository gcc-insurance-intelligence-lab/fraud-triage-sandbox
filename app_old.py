import gradio as gr
import pandas as pd
import random
from datetime import datetime, timedelta

# Fraud detection logic (rule-based for demo)
def calculate_fraud_score(claim_amount, policy_age_days, claim_type, prior_claims, incident_severity):
    """Calculate a fraud risk score based on various factors."""
    score = 0.0
    
    # High claim amount relative to typical
    if claim_amount > 30000:
        score += 0.25
    elif claim_amount > 20000:
        score += 0.15
    
    # New policy (suspicious timing)
    if policy_age_days < 30:
        score += 0.30
    elif policy_age_days < 90:
        score += 0.15
    
    # Multiple prior claims
    if prior_claims >= 3:
        score += 0.25
    elif prior_claims >= 2:
        score += 0.15
    
    # Theft claims are higher risk
    if "Theft" in claim_type:
        score += 0.20
    
    # Severe incidents
    if incident_severity == "Severe":
        score += 0.10
    
    return min(score, 1.0)

def get_risk_level(score):
    """Convert score to risk level."""
    if score >= 0.7:
        return "High Risk"
    elif score >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_recommendation(score):
    """Get triage recommendation based on score."""
    if score >= 0.7:
        return "ESCALATE: Assign to fraud investigation team immediately"
    elif score >= 0.4:
        return "REVIEW: Request additional documentation and verification"
    else:
        return "APPROVE: Proceed with standard claims processing"

def analyze_claim(claim_id, claim_amount, policy_start_date, claim_type, prior_claims, incident_severity):
    """Analyze a claim for fraud indicators."""
    
    # Calculate policy age
    try:
        policy_date = datetime.strptime(policy_start_date, "%Y-%m-%d")
        policy_age_days = (datetime.now() - policy_date).days
    except:
        policy_age_days = 365
    
    # Calculate fraud score
    fraud_score = calculate_fraud_score(
        claim_amount, 
        policy_age_days, 
        claim_type, 
        prior_claims, 
        incident_severity
    )
    
    risk_level = get_risk_level(fraud_score)
    recommendation = get_recommendation(fraud_score)
    
    # Generate detailed analysis
    analysis = f"""
    ### Fraud Triage Analysis
    
    **Claim ID:** {claim_id}
    **Fraud Risk Score:** {fraud_score:.2f} / 1.00
    **Risk Level:** {risk_level}
    
    ---
    
    #### Risk Factors Detected:
    """
    
    factors = []
    
    if claim_amount > 30000:
        factors.append("⚠️ High claim amount (${:,.2f})".format(claim_amount))
    if policy_age_days < 30:
        factors.append("⚠️ Very new policy ({} days old)".format(policy_age_days))
    if prior_claims >= 3:
        factors.append("⚠️ Multiple prior claims ({})".format(prior_claims))
    if "Theft" in claim_type:
        factors.append("⚠️ High-risk claim type (Theft)")
    if incident_severity == "Severe":
        factors.append("⚠️ Severe incident reported")
    
    if factors:
        for factor in factors:
            analysis += f"\n- {factor}"
    else:
        analysis += "\n- ✅ No significant risk factors detected"
    
    analysis += f"""
    
    ---
    
    #### Recommendation:
    
    **{recommendation}**
    
    ---
    
    ⚠️ *This is a demonstration tool using rule-based logic. Real fraud detection requires sophisticated ML models and human oversight.*
    """
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        "Metric": ["Claim ID", "Claim Amount", "Policy Age (days)", "Prior Claims", "Fraud Score", "Risk Level"],
        "Value": [claim_id, f"${claim_amount:,.2f}", policy_age_days, prior_claims, f"{fraud_score:.2f}", risk_level]
    })
    
    return analysis, summary_df

def generate_sample_claim():
    """Generate a random sample claim for testing."""
    claim_types = ["Auto Collision", "Auto Theft", "Home Fire", "Home Water Damage", "Home Burglary"]
    severities = ["Minor", "Moderate", "Severe"]
    
    claim_id = f"CLM-2026-{random.randint(100, 999)}"
    claim_amount = random.choice([5000, 8500, 12000, 18000, 25000, 35000, 50000])
    days_ago = random.randint(1, 730)
    policy_start = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    claim_type = random.choice(claim_types)
    prior_claims = random.randint(0, 4)
    severity = random.choice(severities)
    
    return claim_id, claim_amount, policy_start, claim_type, prior_claims, severity

# Create Gradio interface
with gr.Blocks(title="Fraud Triage Sandbox") as demo:
    gr.Markdown("""
    # 🔍 Fraud Triage Sandbox
    
    **AI-Powered Fraud Detection Demo for Insurance Claims**
    
    This tool demonstrates a rule-based fraud triage system for insurance claims. 
    Enter claim details to receive a fraud risk assessment and triage recommendation.
    
    ⚠️ **DISCLAIMER**: This is a demonstration tool only. All outputs are advisory and for educational purposes.
    """)
    
    with gr.Tab("📝 Manual Entry"):
        gr.Markdown("### Enter claim details for fraud analysis")
        
        with gr.Row():
            with gr.Column():
                claim_id_input = gr.Textbox(label="Claim ID", value="CLM-2026-001", placeholder="e.g., CLM-2026-001")
                claim_amount_input = gr.Number(label="Claim Amount ($)", value=15000, minimum=0)
                policy_date_input = gr.Textbox(label="Policy Start Date (YYYY-MM-DD)", value="2025-12-01")
            
            with gr.Column():
                claim_type_input = gr.Dropdown(
                    choices=["Auto Collision", "Auto Theft", "Auto Vandalism", 
                            "Home Fire", "Home Water Damage", "Home Burglary", "Home Storm Damage"],
                    label="Claim Type",
                    value="Auto Collision"
                )
                prior_claims_input = gr.Slider(label="Prior Claims Count", minimum=0, maximum=10, step=1, value=1)
                severity_input = gr.Radio(choices=["Minor", "Moderate", "Severe"], label="Incident Severity", value="Moderate")
        
        analyze_btn = gr.Button("🔍 Analyze Claim", variant="primary")
        
        analysis_output = gr.Markdown()
        summary_output = gr.Dataframe(label="Claim Summary")
        
        analyze_btn.click(
            fn=analyze_claim,
            inputs=[claim_id_input, claim_amount_input, policy_date_input, claim_type_input, prior_claims_input, severity_input],
            outputs=[analysis_output, summary_output]
        )
    
    with gr.Tab("🎲 Random Sample"):
        gr.Markdown("### Generate and analyze a random sample claim")
        
        generate_btn = gr.Button("🎲 Generate Random Claim", variant="primary")
        
        with gr.Row():
            sample_claim_id = gr.Textbox(label="Claim ID", interactive=False)
            sample_amount = gr.Number(label="Claim Amount ($)", interactive=False)
            sample_policy_date = gr.Textbox(label="Policy Start Date", interactive=False)
        
        with gr.Row():
            sample_claim_type = gr.Textbox(label="Claim Type", interactive=False)
            sample_prior_claims = gr.Number(label="Prior Claims", interactive=False)
            sample_severity = gr.Textbox(label="Severity", interactive=False)
        
        auto_analyze_btn = gr.Button("🔍 Analyze This Claim", variant="secondary")
        
        sample_analysis_output = gr.Markdown()
        sample_summary_output = gr.Dataframe(label="Claim Summary")
        
        generate_btn.click(
            fn=generate_sample_claim,
            outputs=[sample_claim_id, sample_amount, sample_policy_date, sample_claim_type, sample_prior_claims, sample_severity]
        )
        
        auto_analyze_btn.click(
            fn=analyze_claim,
            inputs=[sample_claim_id, sample_amount, sample_policy_date, sample_claim_type, sample_prior_claims, sample_severity],
            outputs=[sample_analysis_output, sample_summary_output]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About Fraud Triage Sandbox
        
        This tool demonstrates a **rule-based fraud detection system** for insurance claims triage.
        
        ### How It Works:
        
        The system evaluates claims based on multiple risk factors:
        
        1. **Claim Amount**: Higher amounts increase risk score
        2. **Policy Age**: Very new policies are flagged (potential fraud timing)
        3. **Prior Claims**: Multiple claims increase suspicion
        4. **Claim Type**: Theft claims carry higher risk
        5. **Incident Severity**: Severe incidents are noted
        
        ### Risk Levels:
        
        - **Low Risk (0.0 - 0.39)**: Standard processing
        - **Medium Risk (0.4 - 0.69)**: Additional verification required
        - **High Risk (0.7 - 1.0)**: Escalate to fraud investigation
        
        ### Limitations:
        
        - This is a **simplified rule-based system** for demonstration
        - Real fraud detection uses sophisticated ML models
        - Human oversight is always required
        - No actual fraud detection should rely solely on automated systems
        
        ### Compliance:
        
        - ✅ All outputs are advisory only
        - ✅ No real claims data used
        - ✅ No actual fraud accusations made
        - ✅ Educational and testing purposes only
        
        ### Use Cases:
        
        - Training claims adjusters
        - Testing fraud detection workflows
        - Educational demonstrations
        - Prototyping triage systems
        """)

if __name__ == "__main__":
    demo.launch()
