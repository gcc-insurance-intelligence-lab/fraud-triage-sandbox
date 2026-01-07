"""
Fraud Detection Utility
Provides core fraud detection logic for the triage sandbox.
"""

import pandas as pd
from typing import Dict, List, Tuple


class FraudDetector:
    """Rule-based fraud detection engine."""
    
    def __init__(self, high_amount_threshold: float = 50000, young_age_threshold: int = 25):
        self.high_amount_threshold = high_amount_threshold
        self.young_age_threshold = young_age_threshold
        
        # High-risk claim types
        self.high_risk_types = [
            "Auto Theft",
            "Property Damage",
            "Liability"
        ]
        
        # High-risk locations
        self.high_risk_locations = [
            "New York",
            "Los Angeles",
            "Miami",
            "Chicago",
            "Houston"
        ]
        
        # Weekend days
        self.weekend_days = ["Saturday", "Sunday"]
    
    def analyze_claim(self, 
                     claim_amount: float,
                     claimant_age: int,
                     claim_type: str,
                     location: str,
                     filing_day: str) -> Dict:
        """
        Analyze a claim for fraud indicators.
        
        Args:
            claim_amount: Dollar amount of the claim
            claimant_age: Age of the claimant
            claim_type: Type of insurance claim
            location: Location where claim was filed
            filing_day: Day of week claim was filed
            
        Returns:
            Dictionary with risk_score, triage_decision, flags, and explanation
        """
        flags = []
        explanations = []
        
        # Check 1: High claim amount
        if claim_amount > self.high_amount_threshold:
            flags.append("High Claim Amount")
            explanations.append(f"Claim amount ${claim_amount:,.2f} exceeds threshold ${self.high_amount_threshold:,.2f}")
        
        # Check 2: Young claimant
        if claimant_age < self.young_age_threshold:
            flags.append("Young Claimant")
            explanations.append(f"Claimant age {claimant_age} is below threshold {self.young_age_threshold}")
        
        # Check 3: High-risk claim type
        if claim_type in self.high_risk_types:
            flags.append("High-Risk Claim Type")
            explanations.append(f"Claim type '{claim_type}' is classified as high-risk")
        
        # Check 4: High-risk location
        if location in self.high_risk_locations:
            flags.append("High-Risk Location")
            explanations.append(f"Location '{location}' has elevated fraud rates")
        
        # Check 5: Weekend filing
        if filing_day in self.weekend_days:
            flags.append("Weekend Filing")
            explanations.append(f"Claim filed on {filing_day}, which is a weekend")
        
        # Calculate risk score
        risk_score = len(flags)
        
        # Determine triage decision
        if risk_score <= 2:
            triage_decision = "Auto-Approve"
            risk_level = "Low Risk"
        elif risk_score <= 4:
            triage_decision = "Manual Review Required"
            risk_level = "Medium Risk"
        else:
            triage_decision = "Flag for Investigation"
            risk_level = "High Risk"
        
        # Build explanation
        if len(flags) == 0:
            explanation = "No fraud indicators detected. Claim appears legitimate."
        else:
            explanation = f"**{risk_level}** - {len(flags)} fraud indicator(s) detected:\n\n"
            for i, exp in enumerate(explanations, 1):
                explanation += f"{i}. {exp}\n"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "triage_decision": triage_decision,
            "flags": flags,
            "explanation": explanation
        }
    
    def batch_analyze(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze multiple claims at once.
        
        Args:
            claims_df: DataFrame with columns: claim_amount, claimant_age, 
                      claim_type, location, filing_day
                      
        Returns:
            DataFrame with added columns: risk_score, triage_decision, flags
        """
        results = []
        
        for _, row in claims_df.iterrows():
            result = self.analyze_claim(
                claim_amount=row['claim_amount'],
                claimant_age=row['claimant_age'],
                claim_type=row['claim_type'],
                location=row['location'],
                filing_day=row['filing_day']
            )
            results.append(result)
        
        # Add results to dataframe
        claims_df['risk_score'] = [r['risk_score'] for r in results]
        claims_df['risk_level'] = [r['risk_level'] for r in results]
        claims_df['triage_decision'] = [r['triage_decision'] for r in results]
        claims_df['flags'] = [', '.join(r['flags']) if r['flags'] else 'None' for r in results]
        
        return claims_df
    
    def get_statistics(self, claims_df: pd.DataFrame) -> Dict:
        """
        Get fraud detection statistics for a batch of claims.
        
        Args:
            claims_df: DataFrame with analyzed claims
            
        Returns:
            Dictionary with statistics
        """
        total_claims = len(claims_df)
        
        if 'triage_decision' not in claims_df.columns:
            claims_df = self.batch_analyze(claims_df)
        
        auto_approve = len(claims_df[claims_df['triage_decision'] == 'Auto-Approve'])
        manual_review = len(claims_df[claims_df['triage_decision'] == 'Manual Review Required'])
        flag_investigation = len(claims_df[claims_df['triage_decision'] == 'Flag for Investigation'])
        
        return {
            'total_claims': total_claims,
            'auto_approve': auto_approve,
            'auto_approve_pct': (auto_approve / total_claims * 100) if total_claims > 0 else 0,
            'manual_review': manual_review,
            'manual_review_pct': (manual_review / total_claims * 100) if total_claims > 0 else 0,
            'flag_investigation': flag_investigation,
            'flag_investigation_pct': (flag_investigation / total_claims * 100) if total_claims > 0 else 0,
            'avg_risk_score': claims_df['risk_score'].mean() if total_claims > 0 else 0
        }
