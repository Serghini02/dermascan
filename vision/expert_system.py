"""
Rule-Based Expert System for DermaScan.
Combines Deep Learning predictions with visual analysis (ABCDE) and symptoms.
"""

from config import DIAGNOSIS_CLASSES, RISK_LEVELS

class MedicalExpertSystem:
    def __init__(self):
        # Configurable thresholds
        self.CONFIDENCE_THRESHOLD = 0.6
        self.ABCDE_HIGH_RISK_THRESHOLD = 7.0
        self.MELANOMA_SENSITIVITY = 0.4  # If melanoma exceeds this, it's suspicious

    def apply_rules(self, cnn_result, abcde_scores, symptoms):
        """
        Applies a set of expert rules to refine the diagnosis.
        """
        rules_triggered = []
        refined_risk = cnn_result.get("risk_level", "benign")
        
        # 1. Consensus Rule (CNN + ABCDE) - Refined
        # If CNN is VERY confident (>80%), we raise the ABCDE suspicion threshold
        cnn_conf = cnn_result.get("confidence", 0.0)
        dynamic_abcde_threshold = self.ABCDE_HIGH_RISK_THRESHOLD
        if cnn_conf > 0.8:
             dynamic_abcde_threshold = 8.5 # Only raise if ABCDE is almost critical
        
        if (cnn_result.get("risk_level") == "benign" and 
            abcde_scores.get("total_score", 0) >= dynamic_abcde_threshold):
            refined_risk = "pre-malignant"
            rules_triggered.append(f"R1: AI/ABCDE Discrepancy. Risk raised by visual morphology (ABCDE {abcde_scores.get('total_score'):.1f}).")

        # 2. Melanoma Rule (Sensitivity)
        # Melanoma is very dangerous; we monitor it with a lower threshold if there is asymmetry
        mel_prob = 0.0
        if "probabilities" in cnn_result:
            mel_info = cnn_result["probabilities"].get("mel", {})
            mel_prob = mel_info.get("probability", 0.0)
            
        if mel_prob > self.MELANOMA_SENSITIVITY and abcde_scores.get("asymmetry", {}).get("score", 0) > 0.7:
            refined_risk = "malignant"
            rules_triggered.append("R2: Suspected Melanoma due to combination of probability and asymmetry.")

        # 3. Critical Symptoms Rule
        # Bleeding is a fundamental clinical indicator
        bleeding = symptoms.get("bleeding", {}).get("positive", False)
        if bleeding and refined_risk != "malignant":
            # If it bleeds and wasn't malignant, it's at least pre-malignant for review
            if refined_risk == "benign":
                refined_risk = "pre-malignant"
            rules_triggered.append("R3: Presence of bleeding (clinical indicator of evolution).")

        # 4. Size Rule (Diameter > 6mm)
        diameter_mm = 0.0
        diameter_detail = abcde_scores.get("diameter", {}).get("detail", "")
        # Try to extract the number from detail "~X.Xmm..."
        try:
            if "~" in diameter_detail:
                diameter_mm = float(diameter_detail.split("~")[1].split("mm")[0])
        except:
            pass

        if diameter_mm > 6.0 and refined_risk == "benign":
            refined_risk = "pre-malignant"
            rules_triggered.append("R4: Diameter greater than 6mm (clinical D criterion).")

        # Determine final recommendation based on refined risk
        diagnosis_name = cnn_result.get("diagnosis_name", "suspicious lesion")
        recommendation, action_priority = self._get_recommendation(refined_risk, abcde_scores, symptoms, diagnosis_name)

        return {
            "refined_risk": refined_risk,
            "risk_label": RISK_LEVELS[refined_risk]["label"],
            "risk_color": RISK_LEVELS[refined_risk]["color"],
            "rules_triggered": rules_triggered,
            "recommendation": recommendation,
            "action_priority": action_priority
        }

    def _get_recommendation(self, risk, abcde, symptoms, diagnosis_name="suspicious lesion"):
        """Generates recommendation based on risk and priorities."""
        abcde_score = abcde.get("total_score", 0)
        positive_symptoms = sum(1 for s in symptoms.values() if isinstance(s, dict) and s.get("positive") is True)

        if risk == "malignant":
            return (f"⚠️ URGENT: Consensus and clinical analysis indicates a HIGH PROBABILITY of {diagnosis_name}. "
                    "You must see a dermatologist immediately for a biopsy. Do not delay this consultation.", "high")
        
        if risk == "pre-malignant":
            # If the risk is pre-malignant but not many positive symptoms, we soften the message
            if positive_symptoms < 2 and abcde_score < 4:
                return ("📋 PREVENTIVE FOLLOW-UP: Some mild atypical features were detected in the analysis. "
                        "Although the risk probability is moderate, it is advised to consult with a dermatologist for a routine check-up.", "medium")
            return ("⚡ IMPORTANT: Atypical features and symptoms requiring professional attention have been detected. "
                    "Schedule an appointment with your specialist for a dermatoscopic review.", "medium")
        
        if abcde_score > 4 or positive_symptoms >= 2:
            return ("🔍 OBSERVATION: Although there are no signs of clear malignancy, there are features that should be monitored. "
                    "We recommend performing a new self-examination in 30 days.", "low")
        
        return ("✅ LOW RISK: The analysis indicates an apparently benign lesion (such as a common mole). "
                "Maintain your regular check-ups and use photoprotection.", "none")
