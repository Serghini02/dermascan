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
        
        # 1. Regla de Consenso (CNN + ABCDE) - Refinada
        # Si la CNN está MUY segura (>80%), elevamos el umbral de sospecha del ABCDE
        cnn_conf = cnn_result.get("confidence", 0.0)
        dynamic_abcde_threshold = self.ABCDE_HIGH_RISK_THRESHOLD
        if cnn_conf > 0.8:
             dynamic_abcde_threshold = 8.5 # Solo subir si el ABCDE es casi crítico
        
        if (cnn_result.get("risk_level") == "benigno" and 
            abcde_scores.get("total_score", 0) >= dynamic_abcde_threshold):
            refined_risk = "pre-maligno"
            rules_triggered.append(f"R1: Discrepancia IA/ABCDE. Riesgo subido por morfología visual (ABCDE {abcde_scores.get('total_score'):.1f}).")

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
        diagnosis_name = cnn_result.get("diagnosis_name", "lesión sospechosa")
        recommendation, action_priority = self._get_recommendation(refined_risk, abcde_scores, symptoms, diagnosis_name)

        return {
            "refined_risk": refined_risk,
            "risk_label": RISK_LEVELS[refined_risk]["label"],
            "risk_color": RISK_LEVELS[refined_risk]["color"],
            "rules_triggered": rules_triggered,
            "recommendation": recommendation,
            "action_priority": action_priority
        }

    def _get_recommendation(self, risk, abcde, symptoms, diagnosis_name="lesión sospechosa"):
        """Genera recomendación basada en riesgo y prioridades."""
        abcde_score = abcde.get("total_score", 0)
        positive_symptoms = sum(1 for s in symptoms.values() if isinstance(s, dict) and s.get("positive") is True)

        if risk == "maligno":
            return (f"⚠️ URGENTE: El análisis de consenso y clínico indica una ALTA PROBABILIDAD de que sea un {diagnosis_name}. "
                    "Debe acudir a un dermatólogo de inmediato para una biopsia. No demore esta consulta.", "alta")
        
        if risk == "pre-maligno":
            # Si el riesgo es pre-maligno pero no hay muchos síntomas positivos, suavizamos el mensaje
            if positive_symptoms < 2 and abcde_score < 4:
                return ("📋 SEGUIMIENTO PREVENTIVO: Se han detectado algunos rasgos atípicos leves en el análisis. "
                        "Aunque la probabilidad de riesgo es moderada, se aconseja consultar con un dermatólogo para una revisión de rutina.", "media")
            return ("⚡ IMPORTANTE: Se han detectado rasgos atípicos y síntomas que requieren atención profesional. "
                    "Programe una cita con su especialista para una revisión dermatoscópica.", "media")
        
        if abcde_score > 4 or positive_symptoms >= 2:
            return ("🔍 OBSERVACIÓN: Aunque no hay signos de malignidad clara, existen rasgos que deben vigilarse. "
                    "Le recomendamos realizar un nuevo autoexamen en 30 días.", "baja")
        
        return ("✅ BAJO RIESGO: El análisis indica una lesión aparentemente benigna (como un lunar común). "
                "Mantenga sus revisiones periódicas habituales y use fotoprotección.", "nula")
