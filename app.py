import gradio as gr
import os
import json
import numpy as np
from openai import OpenAI
from huggingface_hub import hf_hub_download
import joblib

# ML Model Configuration
MODEL_REPO = "gcc-insurance-intelligence-lab/fraud-signal-classifier-v1"
ml_model = None
feature_encoders = None
target_encoder = None

def load_ml_model():
    """Load ML model from local files or Hugging Face Hub"""
    global ml_model, feature_encoders, target_encoder
    
    try:
        # Try local files first (for development/testing)
        local_model_dir = "../fraud-signal-classifier-v1"
        if os.path.exists(f"{local_model_dir}/model.pkl"):
            print("Loading model from local directory...")
            ml_model = joblib.load(f"{local_model_dir}/model.pkl")
            encoders_dict = joblib.load(f"{local_model_dir}/label_encoders.pkl")
            feature_encoders = encoders_dict['feature_encoders']
            target_encoder = encoders_dict['target_encoder']
            print("✓ ML model loaded from local files")
            return True
    except Exception as e:
        print(f"Local model not found: {e}")
    
    try:
        # Download from Hugging Face Hub
        print(f"Downloading model from HF Hub: {MODEL_REPO}...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl")
        encoders_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoders.pkl")
        
        ml_model = joblib.load(model_path)
        encoders_dict = joblib.load(encoders_path)
        feature_encoders = encoders_dict['feature_encoders']
        target_encoder = encoders_dict['target_encoder']
        print("✓ ML model loaded from Hugging Face Hub")
        return True
    except Exception as e:
        print(f"Error loading ML model from HF Hub: {e}")
        return False

# Load model at startup
model_loaded = load_ml_model()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def map_ui_inputs_to_model_features(claim_type, behavior_pattern):
    """Map UI inputs to model feature values"""
    claim_type_map = {
        "Property Damage": "Property Damage",
        "Theft": "Auto Theft",
        "Collision": "Auto Collision",
        "Fire": "Home Fire",
        "Water Damage": "Home Water Damage",
        "Total Loss": "Auto Collision",
        "Liability": "Liability"
    }
    
    behavior_map = {
        "Normal": "Normal",
        "Inconsistent statements": "Inconsistent Details",
        "Rushed filing": "Holiday Filing",
        "Multiple similar claims": "Multiple Claims",
        "Suspicious timing": "Suspicious Timing",
        "Evasive responses": "Suspicious Timing"
    }
    
    policy_type = claim_type_map.get(claim_type, "Property Damage")
    incident_pattern = behavior_map.get(behavior_pattern, "Normal")
    
    return policy_type, incident_pattern

def derive_claimant_risk_from_history(claim_history_count):
    """Derive risk level from claim history"""
    if claim_history_count >= 5:
        return "Very High Risk"
    elif claim_history_count >= 3:
        return "High Risk"
    elif claim_history_count >= 1:
        return "Medium Risk"
    else:
        return "Low Risk"

def analyze_fraud_with_model(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, claim_description):
    """Analyze fraud using ML model. Returns (fraud_score, bucket, predicted_class, probabilities, error_msg)"""
    if not model_loaded or ml_model is None:
        return None, None, None, None, "ML model not available"
    
    try:
        # Map UI inputs to model features
        policy_type, incident_pattern = map_ui_inputs_to_model_features(claim_type, behavior_pattern)
        claimant_profile_risk = derive_claimant_risk_from_history(claim_history_count)
        
        # Convert evidence to document consistency score
        document_consistency_score = evidence_pct / 100.0
        
        # Calculate anomaly score based on risk factors
        anomaly_score = 0.0
        if evidence_pct < 50:
            anomaly_score += 0.3
        if claim_history_count >= 3:
            anomaly_score += 0.3
        if behavior_pattern != "Normal":
            anomaly_score += 0.4
        anomaly_score = min(anomaly_score, 1.0)
        
        # Encode categorical features
        try:
            policy_encoded = feature_encoders['policy_type'].transform([policy_type])[0]
        except:
            policy_encoded = 0
        
        try:
            risk_encoded = feature_encoders['claimant_profile_risk'].transform([claimant_profile_risk])[0]
        except:
            risk_encoded = 0
        
        try:
            pattern_encoded = feature_encoders['incident_pattern'].transform([incident_pattern])[0]
        except:
            pattern_encoded = 0
        
        # Create feature vector
        features = np.array([[
            policy_encoded,
            risk_encoded,
            pattern_encoded,
            document_consistency_score,
            anomaly_score
        ]])
        
        # Get prediction
        probabilities = ml_model.predict_proba(features)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_class = target_encoder.classes_[predicted_idx]
        
        # Calculate fraud score
        fraud_score_map = {
            'Clean': 0.0,
            'Under Review': 0.33,
            'Flagged': 0.66,
            'Confirmed Fraud': 1.0
        }
        
        fraud_score = sum(prob * fraud_score_map.get(label, 0.5) 
                         for prob, label in zip(probabilities, target_encoder.classes_))
        
        # Map to bucket
        if fraud_score < 0.3:
            bucket = "Low"
        elif fraud_score < 0.6:
            bucket = "Medium"
        else:
            bucket = "High"
        
        # Build probability dict
        prob_dict = {label: float(prob) for label, prob in zip(target_encoder.classes_, probabilities)}
        
        return fraud_score, bucket, predicted_class, prob_dict, None
        
    except Exception as e:
        return None, None, None, None, f"ML prediction error: {str(e)}"

def analyze_fraud_with_ai(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, claim_description):
    """Analyze fraud risk using OpenAI GPT-4"""
    if not client:
        return {
            "error": "OpenAI API key not configured.",
            "fallback": True
        }
    
    system_prompt = """You are an expert insurance fraud analyst with 20+ years of experience in the GCC region.
Your role is to provide detailed fraud risk assessments for insurance claims.

IMPORTANT GUIDELINES:
- You provide ADVISORY analysis only - never make final fraud determinations
- All outputs must emphasize human review is mandatory
- Focus on explainability and transparency
- Never accuse anyone of fraud - only identify risk factors

Output your analysis as a JSON object with this structure:
{
    "anomaly_score": <float 0-1>,
    "fraud_likelihood": "<Low|Medium|High>",
    "uncertainty_score": <float 0-1>,
    "risk_factors": [<list of identified risk factors>],
    "protective_factors": [<list of factors that reduce risk>],
    "investigation_recommendations": [<list of recommended actions>],
    "explanation": "<detailed explanation>",
    "confidence_level": "<Low|Medium|High>",
    "requires_immediate_escalation": <boolean>
}"""
    
    user_prompt = f"""Analyze this insurance claim for fraud risk:

**Claim Details:**
- Claim Type: {claim_type}
- Sector: {sector}
- Evidence Provided: {evidence_pct}%
- Behavior Pattern: {behavior_pattern}
- Claim History Count: {claim_history_count} previous claims
- Claim Description: {claim_description}

Provide a comprehensive fraud risk assessment following the JSON structure specified."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        analysis["fallback"] = False
        return analysis
        
    except Exception as e:
        return {
            "error": f"AI analysis failed: {str(e)}",
            "fallback": True
        }

def calculate_fraud_likelihood(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count):
    """Calculate fraud likelihood based on rule-based logic"""
    anomaly_score = 0.0
    
    if evidence_pct < 30:
        anomaly_score += 0.35
    elif evidence_pct < 50:
        anomaly_score += 0.20
    elif evidence_pct < 70:
        anomaly_score += 0.10
    
    if behavior_pattern == "Inconsistent statements":
        anomaly_score += 0.25
    elif behavior_pattern == "Rushed filing":
        anomaly_score += 0.20
    elif behavior_pattern == "Multiple similar claims":
        anomaly_score += 0.30
    elif behavior_pattern == "Suspicious timing":
        anomaly_score += 0.15
    
    if claim_history_count >= 5:
        anomaly_score += 0.25
    elif claim_history_count >= 3:
        anomaly_score += 0.15
    elif claim_history_count >= 2:
        anomaly_score += 0.08
    
    high_risk_types = ["Theft", "Total Loss", "Fire"]
    if claim_type in high_risk_types:
        anomaly_score += 0.15
    
    if sector == "Commercial":
        anomaly_score += 0.05
    
    anomaly_score = min(anomaly_score, 1.0)
    return anomaly_score

def get_fraud_bucket(anomaly_score):
    """Classify fraud likelihood into buckets"""
    if anomaly_score >= 0.65:
        return "High"
    elif anomaly_score >= 0.35:
        return "Medium"
    else:
        return "Low"

def calculate_uncertainty(evidence_pct, claim_history_count):
    """Calculate uncertainty score"""
    uncertainty = 0.0
    
    if evidence_pct < 40:
        uncertainty += 0.40
    elif evidence_pct < 60:
        uncertainty += 0.25
    elif evidence_pct < 80:
        uncertainty += 0.15
    
    if claim_history_count == 0:
        uncertainty += 0.30
    elif claim_history_count == 1:
        uncertainty += 0.15
    
    return min(uncertainty, 1.0)

def calculate_fraud_likelihood_fallback(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count):
    """Fallback rule-based fraud detection"""
    anomaly_score = calculate_fraud_likelihood(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count)
    bucket = get_fraud_bucket(anomaly_score)
    uncertainty = calculate_uncertainty(evidence_pct, claim_history_count)
    
    return {
        "anomaly_score": anomaly_score,
        "fraud_likelihood": bucket,
        "uncertainty_score": uncertainty,
        "fallback": True
    }

def format_analysis_output(analysis, claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, 
                          ml_score, ml_bucket, ml_class, ml_probs):
    """Format analysis with hybrid ML + rule-based results"""
    
    if analysis.get("fallback"):
        mode = "⚙️ Rule-Based Mode (AI Unavailable)"
        if "error" in analysis:
            error_msg = f"\n\n⚠️ **Note:** {analysis['error']}\n\nFalling back to rule-based analysis.\n"
        else:
            error_msg = ""
    else:
        mode = "🤖 AI-Powered Analysis"
        error_msg = ""
    
    ml_available = ml_score is not None
    
    if ml_available:
        mode = "🔬 Hybrid Mode: AI + ML Model"
    
    output = f"# {mode}\n{error_msg}\n"
    output += f"## Fraud Triage Assessment\n\n"
    output += f"**Claim Type:** {claim_type}\n"
    output += f"**Sector:** {sector}\n"
    output += f"**Evidence Provided:** {evidence_pct}%\n"
    output += f"**Behavior Pattern:** {behavior_pattern}\n"
    output += f"**Claim History Count:** {claim_history_count}\n\n"
    output += f"---\n\n"
    
    # Rule-based metrics
    output += f"### 📊 Risk Metrics (Rule-Based)\n\n"
    output += f"- **Anomaly Score:** {analysis['anomaly_score']:.2f} / 1.00\n"
    output += f"- **Fraud Likelihood:** {analysis['fraud_likelihood']}\n"
    output += f"- **Uncertainty Score:** {analysis.get('uncertainty_score', 0):.2f} / 1.00\n"
    
    if not analysis.get("fallback"):
        output += f"- **Confidence Level:** {analysis.get('confidence_level', 'N/A')}\n"
        output += f"- **Immediate Escalation Required:** {'Yes' if analysis.get('requires_immediate_escalation') else 'No'}\n"
    
    # ML model metrics
    if ml_available:
        output += f"\n### 🤖 ML Model Metrics\n\n"
        output += f"- **ML Fraud Score:** {ml_score:.3f} / 1.00\n"
        output += f"- **ML Bucket:** {ml_bucket}\n"
        output += f"- **ML Predicted Class:** {ml_class}\n"
        
        if ml_probs:
            output += f"\n**Class Probabilities:**\n"
            for label, prob in ml_probs.items():
                output += f"  - {label}: {prob:.3f}\n"
        
        # Hybrid decision
        rule_bucket = analysis['fraud_likelihood']
        output += f"\n### ⚖️ Hybrid Decision\n\n"
        output += f"- **Rule-Based Bucket:** {rule_bucket}\n"
        output += f"- **ML Model Bucket:** {ml_bucket}\n"
        
        bucket_priority = {'Low': 1, 'Medium': 2, 'High': 3}
        if bucket_priority.get(ml_bucket, 0) > bucket_priority.get(rule_bucket, 0):
            final_bucket = ml_bucket
            escalation = True
            output += f"- **Final Bucket:** {final_bucket} ⬆️ (Escalated by ML model)\n"
        elif bucket_priority.get(rule_bucket, 0) > bucket_priority.get(ml_bucket, 0):
            final_bucket = rule_bucket
            escalation = True
            output += f"- **Final Bucket:** {final_bucket} ⬆️ (Escalated by rules)\n"
        else:
            final_bucket = rule_bucket
            escalation = False
            output += f"- **Final Bucket:** {final_bucket} ✓ (Agreement)\n"
        
        if escalation:
            output += f"\n⚠️ **Disagreement Detected:** ML and rule-based systems produced different risk levels. Taking higher severity for safety.\n"
    
    output += f"\n---\n\n"
    
    # Risk factors
    if not analysis.get("fallback") and "risk_factors" in analysis:
        output += f"### ⚠️ Risk Factors Identified\n\n"
        for factor in analysis["risk_factors"]:
            output += f"- {factor}\n"
        output += f"\n"
    
    # Protective factors
    if not analysis.get("fallback") and "protective_factors" in analysis:
        output += f"### ✅ Protective Factors\n\n"
        for factor in analysis["protective_factors"]:
            output += f"- {factor}\n"
        output += f"\n"
    
    # Explanation
    if not analysis.get("fallback") and "explanation" in analysis:
        output += f"### 📝 Detailed Explanation\n\n"
        output += f"{analysis['explanation']}\n\n"
    
    # Investigation recommendations
    if not analysis.get("fallback") and "investigation_recommendations" in analysis:
        output += f"### 🔍 Investigation Recommendations\n\n"
        for i, rec in enumerate(analysis["investigation_recommendations"], 1):
            output += f"{i}. {rec}\n"
        output += f"\n"
    
    # Human review warning
    output += f"---\n\n"
    output += f"### ⚠️ MANDATORY HUMAN REVIEW\n\n"
    
    bucket = final_bucket if ml_available and 'final_bucket' in locals() else analysis["fraud_likelihood"]
    if bucket == "High":
        output += "**CRITICAL:** This claim has been flagged as HIGH RISK for potential fraud. "
        output += "Immediate human investigation is required before any processing decisions are made.\n\n"
    elif bucket == "Medium":
        output += "**CAUTION:** This claim shows moderate fraud indicators. "
        output += "Enhanced review and verification are required.\n\n"
    else:
        output += "**STANDARD:** This claim shows low fraud indicators, but human review is still required. "
        output += "Follow standard claims processing procedures.\n\n"
    
    output += "**IMPORTANT:** This is an advisory tool only. All fraud determinations must be made by qualified human investigators. "
    output += "No automated system should make final decisions on fraud accusations or claim denials.\n\n"
    
    output += "**Governance:** This analysis is logged for audit purposes and must be reviewed by a licensed fraud investigator before any action is taken."
    
    return output

def analyze_fraud_risk(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, claim_description):
    """Main function to analyze fraud risk with AI, ML model, and rule-based fallback"""
    
    # Get ML model prediction
    ml_score, ml_bucket, ml_class, ml_probs, ml_error = analyze_fraud_with_model(
        claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, claim_description
    )
    
    # Try AI analysis
    analysis = analyze_fraud_with_ai(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, claim_description)
    
    # If AI fails, use fallback
    if analysis.get("fallback") and "error" in analysis:
        fallback_analysis = calculate_fraud_likelihood_fallback(claim_type, sector, evidence_pct, behavior_pattern, claim_history_count)
        fallback_analysis["error"] = analysis["error"]
        analysis = fallback_analysis
    
    # Format output with ML results
    detailed_output = format_analysis_output(
        analysis, claim_type, sector, evidence_pct, behavior_pattern, claim_history_count,
        ml_score, ml_bucket, ml_class, ml_probs
    )
    
    # Create summary
    rule_bucket = analysis['fraud_likelihood']
    
    summary = f"""**Fraud Likelihood (Rules):** {rule_bucket}
**Anomaly Score:** {analysis['anomaly_score']:.2f}
**Uncertainty Score:** {analysis.get('uncertainty_score', 0):.2f}"""
    
    if ml_score is not None:
        summary += f"\n**ML Fraud Score:** {ml_score:.3f}"
        summary += f"\n**ML Bucket:** {ml_bucket}"
        summary += f"\n**ML Predicted Class:** {ml_class}"
        
        bucket_priority = {'Low': 1, 'Medium': 2, 'High': 3}
        if bucket_priority.get(ml_bucket, 0) != bucket_priority.get(rule_bucket, 0):
            final_bucket = ml_bucket if bucket_priority.get(ml_bucket, 0) > bucket_priority.get(rule_bucket, 0) else rule_bucket
            summary += f"\n**Final Bucket:** {final_bucket} (Escalated)"
        else:
            summary += f"\n**Final Bucket:** {rule_bucket} (Agreement)"
    else:
        summary += f"\n**ML Model:** {ml_error if ml_error else 'Not available'}"
    
    summary += f"\n**Human Review:** Required"
    summary += f"\n**Mode:** {'Hybrid (AI + ML)' if ml_score and not analysis.get('fallback') else 'AI-Powered' if not analysis.get('fallback') else 'Rule-Based'}"
    
    return detailed_output, summary, analysis['anomaly_score'], analysis.get('uncertainty_score', 0)

# Create Gradio interface
with gr.Blocks(title="Fraud Triage Sandbox - Hybrid Edition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Fraud Triage Sandbox - Hybrid Edition
    
    **Hybrid Fraud Detection: AI + ML Model + Rule-Based Logic**
    
    This tool combines:
    - 🤖 OpenAI GPT-4 for intelligent fraud analysis
    - 🔬 ML Model (fraud-signal-classifier-v1) for pattern detection
    - ⚙️ Rule-based logic for baseline assessment
    
    ## ⚠️ MANDATORY DISCLAIMER
    
    **This is a demonstration tool for educational purposes only.**
    
    - ✅ All outputs are **advisory only** and require human verification
    - ✅ No AI/ML component issues approvals, denials, or financial outcomes
    - ✅ All fraud determinations must be made by qualified human investigators
    - ✅ Hybrid mode combines multiple signals for enhanced detection
    - ✅ Not for use in actual claim approval, pricing, or reserving decisions
    
    **Human-in-the-loop is mandatory for all fraud investigations.**
    
    ## 🔐 Configuration
    
    To enable AI analysis, set the `OPENAI_API_KEY` environment variable or Hugging Face Secret.
    ML model is loaded automatically from Hugging Face Hub.
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
            
            claim_description = gr.Textbox(
                label="Claim Description",
                placeholder="Describe the claim incident in detail...",
                lines=5,
                value="Vehicle collision at intersection. Driver claims other party ran red light. Minor damage to front bumper and headlight."
            )
            
            analyze_btn = gr.Button("🔍 Analyze Fraud Risk", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### Assessment Results")
            
            summary_output = gr.Textbox(
                label="Quick Summary",
                lines=8,
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
        inputs=[claim_type, sector, evidence_pct, behavior_pattern, claim_history_count, claim_description],
        outputs=[detailed_output, summary_output, anomaly_output, uncertainty_output]
    )
    
    with gr.Accordion("ℹ️ About This Tool", open=False):
        gr.Markdown("""
        ## How It Works
        
        This fraud triage system uses **Hybrid Detection** combining three approaches:
        
        ### 1. AI Analysis (OpenAI GPT-4)
        - Natural language understanding of claim descriptions
        - Contextual risk factor identification
        - Investigation recommendations
        
        ### 2. ML Model (fraud-signal-classifier-v1)
        - Trained on synthetic insurance fraud data
        - Pattern recognition across multiple features
        - Probability-based risk scoring
        - Bucket classification (Low/Medium/High)
        
        ### 3. Rule-Based Logic
        - Evidence completeness assessment
        - Behavior pattern evaluation
        - Claim history analysis
        - Baseline fraud indicators
        
        ### Hybrid Decision Logic
        - Combines all three signals
        - Takes higher severity when systems disagree
        - Flags disagreements for human review
        - Provides comprehensive risk assessment
        
        ### Fraud Likelihood Buckets
        - **Low (0.0 - 0.34)**: Standard processing
        - **Medium (0.35 - 0.64)**: Enhanced review
        - **High (0.65 - 1.0)**: Immediate investigation
        
        ### Use Cases
        - Training and education
        - Workflow prototyping
        - AI/ML demonstration
        - Fraud detection research
        
        **Built for GCC Insurance Intelligence Lab**
        """
        )

if __name__ == "__main__":
    demo.launch()
