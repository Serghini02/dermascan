"""
Sistema Experto Basado en Reglas para DermaScan.
Combina predicciones de Deep Learning con análisis visual (ABCDE) y síntomas.
"""

from config import DIAGNOSIS_CLASSES, RISK_LEVELS

class MedicalExpertSystem:
    def __init__(self):
        # Umbrales configurables
        self.CONFIDENCE_THRESHOLD = 0.6
        self.ABCDE_HIGH_RISK_THRESHOLD = 7.0
        self.MELANOMA_SENSITIVITY = 0.4  # Si el melanoma supera esto, ya es sospechoso

    def apply_rules(self, cnn_result, abcde_scores, symptoms):
        """
        Aplica un conjunto de reglas expertas para refinar el diagnóstico.
        """
        rules_triggered = []
        refined_risk = cnn_result.get("risk_level", "benigno")
        
        # 1. Regla de Consenso (CNN + ABCDE)
        # Si la CNN dice benigno pero ABCDE dice riesgo alto, subir a pre-maligno
        if (cnn_result.get("risk_level") == "benigno" and 
            abcde_scores.get("total_score", 0) >= self.ABCDE_HIGH_RISK_THRESHOLD):
            refined_risk = "pre-maligno"
            rules_triggered.append("R1: Discrepancia entre IA y morfología visual (ABCDE alto).")

        # 2. Regla de Melanoma (Sensibilidad)
        # El melanoma es muy peligroso, lo vigilamos con menor umbral si hay asimetría
        mel_prob = 0.0
        if "probabilities" in cnn_result:
            mel_info = cnn_result["probabilities"].get("mel", {})
            mel_prob = mel_info.get("probability", 0.0)
            
        if mel_prob > self.MELANOMA_SENSITIVITY and abcde_scores.get("asymmetry", {}).get("score", 0) > 0.5:
            refined_risk = "maligno"
            rules_triggered.append("R2: Sospecha de Melanoma por combinación de probabilidad y asimetría.")

        # 3. Regla de Síntomas Críticos
        # El sangrado es un indicador clínico fundamental
        sangrado = symptoms.get("sangrado", {}).get("positive", False)
        if sangrado and refined_risk != "maligno":
            # Si sangra y no era maligno, al menos es pre-maligno para revisión
            if refined_risk == "benigno":
                refined_risk = "pre-maligno"
            rules_triggered.append("R3: Presencia de sangrado (indicador clínico de evolución).")

        # 4. Regla de Tamaño (Diámetro > 6mm)
        diameter_mm = 0.0
        diameter_detail = abcde_scores.get("diameter", {}).get("detail", "")
        # Intentar extraer el número del detail "~X.Xmm..."
        try:
            if "~" in diameter_detail:
                diameter_mm = float(diameter_detail.split("~")[1].split("mm")[0])
        except:
            pass

        if diameter_mm > 6.0 and refined_risk == "benigno":
            refined_risk = "pre-maligno"
            rules_triggered.append("R4: Diámetro mayor a 6mm (criterio clínico D).")

        # Determinar recomendación final basada en el riesgo refinado
        recommendation, action_priority = self._get_recommendation(refined_risk, abcde_scores, symptoms)

        return {
            "refined_risk": refined_risk,
            "risk_label": RISK_LEVELS[refined_risk]["label"],
            "risk_color": RISK_LEVELS[refined_risk]["color"],
            "rules_triggered": rules_triggered,
            "recommendation": recommendation,
            "action_priority": action_priority
        }

    def _get_recommendation(self, risk, abcde, symptoms):
        """Genera recomendación basada en riesgo y prioridades."""
        abcde_score = abcde.get("total_score", 0)
        positive_symptoms = sum(1 for s in symptoms.values() if isinstance(s, dict) and s.get("positive") is True)

        if risk == "maligno":
            return ("⚠️ URGENTE: Los indicadores sugieren una lesión de alto riesgo. "
                    "Se recomienda acudir a un dermatólogo en las próximas 24-48 horas para una biopsia.", "alta")
        
        if risk == "pre-maligno":
            return ("⚡ IMPORTANTE: Se han detectado rasgos atípicos. "
                    "Programe una cita con su especialista para una revisión dermatoscópica.", "media")
        
        if abcde_score > 4 or positive_symptoms >= 2:
            return ("📋 SEGUIMIENTO: Aunque los modelos no indican malignidad clara, hay rasgos que vigilar. "
                    "Repita el escaneo en 4 semanas y observe cambios.", "baja")
        
        return ("✅ BAJO RIESGO: Los indicadores están dentro de la normalidad. "
                "Continúe con sus revisiones habituales y use protección solar.", "nula")
