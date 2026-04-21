"""
=============================================================================
EVALUACION DEL EXTRACTOR DE SINTOMAS -- DermaScan
=============================================================================
Este script construye un dataset de prueba con frases reales de pacientes,
las pasa por `extract_symptoms()` y mide si el sistema "acierta" o no.

Métricas calculadas por síntoma y globales:
  - Precision, Recall, F1-Score
  - Exactitud por caso (porcentaje de síntomas bien extraídos)
  - Informe de errores detallado (falsos positivos / negativos)

Uso:
    python evaluation/evaluate_symptom_extractor.py
=============================================================================
"""

import sys
import os
import json

# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Aseguramos que el módulo nlp sea importable desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.symptom_extractor import extract_symptoms

# =============================================================================
# DATASET DE EVALUACIÓN
# Cada caso representa una frase que el paciente podría decir como respuesta
# a UNA pregunta. El campo "expected" contiene el valor esperado para cada
# síntoma: True (positivo), False (negativo) o None (no mencionado/irrelevante).
# =============================================================================
EVALUATION_DATASET = [
    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 1: DOLOR (Total: 16)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "D-01", "question": "dolor", "text": "Sí, me duele bastante cuando lo toco.", "expected": {"dolor": True} },
    { "id": "D-02", "question": "dolor", "text": "No me duele para nada, no siento nada.", "expected": {"dolor": False} },
    { "id": "D-03", "question": "dolor", "text": "Tengo algo de molestia cuando presiono la zona.", "expected": {"dolor": True} },
    { "id": "D-04", "question": "dolor", "text": "A veces arde un poco, sobre todo por las noches.", "expected": {"dolor": True} },
    { "id": "D-05", "question": "dolor", "text": "No siento nada especial, está ahí pero sin dolor.", "expected": {"dolor": False} },
    { "id": "D-06", "question": "dolor", "text": "Para nada, no hay ningún tipo de dolor.", "expected": {"dolor": False} },
    { "id": "D-07", "question": "dolor", "text": "Me escuece un poco si me pongo ropa ajustada.", "expected": {"dolor": True} },
    { "id": "D-08", "question": "dolor", "text": "Me da pinchazos fuertes de vez en cuando.", "expected": {"dolor": True} },
    { "id": "D-09", "question": "dolor", "text": "No experimento ninguna molestia al presionar.", "expected": {"dolor": False} },
    { "id": "D-10", "question": "dolor", "text": "Es un dolor sordo pero constante.", "expected": {"dolor": True} },
    { "id": "D-11", "question": "dolor", "text": "Siento como que me quema la piel por esa zona.", "expected": {"dolor": True} },
    { "id": "D-12", "question": "dolor", "text": "Afortunadamente no me duele nada.", "expected": {"dolor": False} },
    { "id": "D-13", "question": "dolor", "text": "Está muy sensible al roce, duele.", "expected": {"dolor": True} },
    { "id": "D-14", "question": "dolor", "text": "No me ha dolido nunca, ni siquiera al principio.", "expected": {"dolor": False} },
    { "id": "D-15", "question": "dolor", "text": "Siento un latido doloroso en el lunar.", "expected": {"dolor": True} },
    { "id": "D-16", "question": "dolor", "text": "Nada de dolor, cero molestias.", "expected": {"dolor": False} },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 2: PICOR (Total: 16)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "P-01", "question": "picor", "text": "Sí, me pica mucho, especialmente por la tarde.", "expected": {"picor": True} },
    { "id": "P-02", "question": "picor", "text": "No me pica, nunca he notado picor.", "expected": {"picor": False} },
    { "id": "P-03", "question": "picor", "text": "Tengo comezón de vez en cuando, no siempre.", "expected": {"picor": True} },
    { "id": "P-04", "question": "picor", "text": "A veces pica un poco, sí.", "expected": {"picor": True} },
    { "id": "P-05", "question": "picor", "text": "Sin picor, todo normal en esa zona.", "expected": {"picor": False} },
    { "id": "P-06", "question": "picor", "text": "La verdad es que no me rasco nunca, no pica.", "expected": {"picor": False} },
    { "id": "P-07", "question": "picor", "text": "Siento una necesidad imperiosa de rascarme.", "expected": {"picor": True} },
    { "id": "P-08", "question": "picor", "text": "No tengo comezón, está muy tranquila la zona.", "expected": {"picor": False} },
    { "id": "P-09", "question": "picor", "text": "Me pica horrores por la noche.", "expected": {"picor": True} },
    { "id": "P-10", "question": "picor", "text": "Es un picor insoportable a ratos.", "expected": {"picor": True} },
    { "id": "P-11", "question": "picor", "text": "No noto que me pique en absoluto.", "expected": {"picor": False} },
    { "id": "P-12", "question": "picor", "text": "Siento como hormigueo y picor constante.", "expected": {"picor": True} },
    { "id": "P-13", "question": "picor", "text": "Sin rastro de picor ni escozor.", "expected": {"picor": False} },
    { "id": "P-14", "question": "picor", "text": "Me dan ganas de rascarme todo el tiempo.", "expected": {"picor": True} },
    { "id": "P-15", "question": "picor", "text": "Ni me pica ni me molesta.", "expected": {"picor": False} },
    { "id": "P-16", "question": "picor", "text": "Me pica bastante si sudo mucho.", "expected": {"picor": True} },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 3: TAMAÑO (Total: 16)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "T-01", "question": "tamaño", "text": "Sí, ha crecido bastante en los últimos meses.", "expected": {"tamaño": True} },
    { "id": "T-02", "question": "tamaño", "text": "No, sigue siendo del mismo tamaño que siempre.", "expected": {"tamaño": False} },
    { "id": "T-03", "question": "tamaño", "text": "Me parece que está más grande que antes.", "expected": {"tamaño": True} },
    { "id": "T-04", "question": "tamaño", "text": "Creo que no ha cambiado, lo veo igual que siempre.", "expected": {"tamaño": False} },
    { "id": "T-05", "question": "tamaño", "text": "Ha aumentado de tamaño notablemente este año.", "expected": {"tamaño": True} },
    { "id": "T-06", "question": "tamaño", "text": "Era más pequeño cuando era niño, ahora es más grande.", "expected": {"tamaño": True} },
    { "id": "T-07", "question": "tamaño", "text": "Es idéntico a cuando me salió, no ha variado nada.", "expected": {"tamaño": False} },
    { "id": "T-08", "question": "tamaño", "text": "Se ha vuelto mucho más abultado y grande.", "expected": {"tamaño": True} },
    { "id": "T-09", "question": "tamaño", "text": "Sigue exactamente igual que hace diez años.", "expected": {"tamaño": False} },
    { "id": "T-10", "question": "tamaño", "text": "Noto que los bordes se están expandiendo.", "expected": {"tamaño": True} },
    { "id": "T-11", "question": "tamaño", "text": "Ha duplicado su tamaño en apenas un mes.", "expected": {"tamaño": True} },
    { "id": "T-12", "question": "tamaño", "text": "No ha crecido ni un milímetro, está igual.", "expected": {"tamaño": False} },
    { "id": "T-13", "question": "tamaño", "text": "Antes era un punto pequeño y ahora es una mancha.", "expected": {"tamaño": True} },
    { "id": "T-14", "question": "tamaño", "text": "Su tamaño es constante, no ha variado nada.", "expected": {"tamaño": False} },
    { "id": "T-15", "question": "tamaño", "text": "Creo que está creciendo hacia fuera.", "expected": {"tamaño": True} },
    { "id": "T-16", "question": "tamaño", "text": "No ha cambiado de forma ni de tamaño.", "expected": {"tamaño": False} },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 4: SANGRADO (Total: 16)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "S-01", "question": "sangrado", "text": "Sí, una vez sangró cuando me lo rasqué.", "expected": {"sangrado": True} },
    { "id": "S-02", "question": "sangrado", "text": "No, nunca ha sangrado en ningún momento.", "expected": {"sangrado": False} },
    { "id": "S-03", "question": "sangrado", "text": "A veces sangra un poquito, sin causa aparente.", "expected": {"sangrado": True} },
    { "id": "S-04", "question": "sangrado", "text": "No ningún sangrado, está completamente seco.", "expected": {"sangrado": False} },
    { "id": "S-05", "question": "sangrado", "text": "Ha sangrado dos veces estas semanas.", "expected": {"sangrado": True, "duracion": True} },
    { "id": "S-06", "question": "sangrado", "text": "Jamás ha soltado sangre ni nada parecido.", "expected": {"sangrado": False} },
    { "id": "S-07", "question": "sangrado", "text": "Amanecí con la sábana manchada de sangre.", "expected": {"sangrado": True} },
    { "id": "S-08", "question": "sangrado", "text": "No sangra aunque se me enganche con la ropa.", "expected": {"sangrado": False} },
    { "id": "S-09", "question": "sangrado", "text": "Suelto un poco de líquido con sangre a veces.", "expected": {"sangrado": True} },
    { "id": "S-10", "question": "sangrado", "text": "Se le ha hecho una costra de sangre.", "expected": {"sangrado": True} },
    { "id": "S-11", "question": "sangrado", "text": "Ningún episodio de sangrado hasta la fecha.", "expected": {"sangrado": False} },
    { "id": "S-12", "question": "sangrado", "text": "Sangró espontáneamente ayer por la tarde.", "expected": {"sangrado": True} },
    { "id": "S-13", "question": "sangrado", "text": "Está muy seco, no sangra nunca.", "expected": {"sangrado": False} },
    { "id": "S-14", "question": "sangrado", "text": "Noto que supura un poco de sangre.", "expected": {"sangrado": True} },
    { "id": "S-15", "question": "sangrado", "text": "No ha sangrado ni una sola vez.", "expected": {"sangrado": False} },
    { "id": "S-16", "question": "sangrado", "text": "A veces mancho el apósito de sangre.", "expected": {"sangrado": True} },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 5: COLOR (Total: 16)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "C-01", "question": "color", "text": "Sí, se ha oscurecido bastante, antes era más claro.", "expected": {"color": True} },
    { "id": "C-02", "question": "color", "text": "No ha cambiado de color, sigue igual.", "expected": {"color": False} },
    { "id": "C-03", "question": "color", "text": "Tiene colores diferentes, como manchas dentro del lunar.", "expected": {"color": True} },
    { "id": "C-04", "question": "color", "text": "El color es el mismo de siempre, uniforme y marrón.", "expected": {"color": False} },
    { "id": "C-05", "question": "color", "text": "Cambió de color, ahora tiene partes rojizas.", "expected": {"color": True} },
    { "id": "C-06", "question": "color", "text": "Está mucho más negro que al principio.", "expected": {"color": True} },
    { "id": "C-07", "question": "color", "text": "No noto ninguna variación cromática, está igual.", "expected": {"color": False} },
    { "id": "C-08", "question": "color", "text": "Tiene un borde azulado que antes no tenía.", "expected": {"color": True} },
    { "id": "C-09", "question": "color", "text": "Mantiene su tono marrón uniforme de siempre.", "expected": {"color": False} },
    { "id": "C-10", "question": "color", "text": "Se ha vuelto bicolor, marrón y negro.", "expected": {"color": True} },
    { "id": "C-11", "question": "color", "text": "Están apareciendo zonas blancas dentro.", "expected": {"color": True} },
    { "id": "C-12", "question": "color", "text": "No ha habido ningún cambio en la pigmentación.", "expected": {"color": False} },
    { "id": "C-13", "question": "color", "text": "Está mucho más rojizo que la semana pasada.", "expected": {"color": True} },
    { "id": "C-14", "question": "color", "text": "El color es el mismo, no ha mutado.", "expected": {"color": False} },
    { "id": "C-15", "question": "color", "text": "Se ha puesto muy oscuro, casi negro azabache.", "expected": {"color": True} },
    { "id": "C-16", "question": "color", "text": "Su tonalidad sigue siendo la misma.", "expected": {"color": False} },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 6: DURACIÓN (Total: 16)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "DU-01", "question": "duracion", "text": "Lo tengo desde hace unos 3 meses.", "expected": {"duracion": True} },
    { "id": "DU-02", "question": "duracion", "text": "Lo empecé a notar hace 2 años aproximadamente.", "expected": {"duracion": True} },
    { "id": "DU-03", "question": "duracion", "text": "Lo tengo de toda la vida, desde que nací.", "expected": {"duracion": True} },
    { "id": "DU-04", "question": "duracion", "text": "Recientemente noté los cambios, hace unas semanas.", "expected": {"duracion": True} },
    { "id": "DU-05", "question": "duracion", "text": "No lo sé exactamente, hace bastante tiempo.", "expected": {"duracion": True} },
    { "id": "DU-06", "question": "duracion", "text": "Apareció hace apenas unos días.", "expected": {"duracion": True} },
    { "id": "DU-07", "question": "duracion", "text": "Me salió hace cuestión de quince días.", "expected": {"duracion": True} },
    { "id": "DU-08", "question": "duracion", "text": "Llevo con esto más de una década.", "expected": {"duracion": True} },
    { "id": "DU-09", "question": "duracion", "text": "Apareció repentinamente el verano pasado.", "expected": {"duracion": True} },
    { "id": "DU-10", "question": "duracion", "text": "Hace unos cinco años que lo tengo.", "expected": {"duracion": True} },
    { "id": "DU-11", "question": "duracion", "text": "Es algo nuevo de este semestre.", "expected": {"duracion": True} },
    { "id": "DU-12", "question": "duracion", "text": "Lo descubrí hace tan solo tres semanas.", "expected": {"duracion": True} },
    { "id": "DU-13", "question": "duracion", "text": "Lo tengo ahí desde que tengo memoria.", "expected": {"duracion": True} },
    { "id": "DU-14", "question": "duracion", "text": "Apenas lleva conmigo un par de meses.", "expected": {"duracion": True} },
    { "id": "DU-15", "question": "duracion", "text": "Salió hace mucho tiempo, no recuerdo cuándo.", "expected": {"duracion": True} },
    { "id": "DU-16", "question": "duracion", "text": "Ha aparecido en el último año.", "expected": {"duracion": True} },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 7: CASOS MIXTOS (Total: 6)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "MX-01", "question": "general", "text": "Me pica y también ha cambiado de color, está más oscuro.", "expected": {"picor": True, "color": True} },
    { "id": "MX-02", "question": "general", "text": "Duele un poco y sangró el otro día.", "expected": {"dolor": True, "sangrado": True} },
    { "id": "MX-03", "question": "general", "text": "No duele, no pica y no ha cambiado de tamaño.", "expected": {"dolor": False, "picor": False, "tamaño": False} },
    { "id": "MX-04", "question": "general", "text": "Creció bastante y lo noto desde hace 6 semanas.", "expected": {"tamaño": True, "duracion": True} },
    { "id": "MX-05", "question": "general", "text": "No siento dolor ni picor, pero sí noto que ha crecido mucho.", "expected": {"dolor": False, "picor": False, "tamaño": True} },
    { "id": "MX-06", "question": "general", "text": "Está más grande, más oscuro y además me duele.", "expected": {"tamaño": True, "color": True, "dolor": True} },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 8: RESPUESTAS CORTAS (Total: 4)
    # ─────────────────────────────────────────────────────────────────────────
    { "id": "SC-01", "question": "dolor", "text": "Sí.", "expected": {"dolor": True} },
    { "id": "SC-02", "question": "dolor", "text": "No.", "expected": {"dolor": False} },
    { "id": "SC-03", "question": "picor", "text": "Un poquito nada más.", "expected": {"picor": True} },
    { "id": "SC-04", "question": "sangrado", "text": "Nunca.", "expected": {"sangrado": False} },
]

# =============================================================================
# FUNCIONES DE EVALUACIÓN
# =============================================================================

SYMPTOM_KEYS = ["dolor", "picor", "tamaño", "sangrado", "color", "duracion"]


def evaluate_case(case):
    """
    Evalúa un único caso del dataset.
    Retorna un dict con:
      - extracted: resultado bruto del extractor
      - per_symptom: {sintoma: {expected, got, correct}}
      - case_accuracy: fracción de síntomas evaluables correctamente detectados
    """
    # Pasamos el contexto de la pregunta para mejorar detección de respuestas cortas
    ctx = case["question"] if case["question"] in SYMPTOM_KEYS else None
    extracted = extract_symptoms(case["text"], context_symptom=ctx)
    expected  = case["expected"]

    per_symptom = {}
    evaluated   = 0
    correct     = 0

    for sym in SYMPTOM_KEYS:
        exp_val = expected.get(sym)
        got_val = extracted.get(sym, {}).get("positive")

        # Solo evaluamos síntomas con expectativa definida (no None)
        if exp_val is None:
            per_symptom[sym] = {"expected": None, "got": got_val, "correct": None, "skip": True}
            continue

        is_correct = (got_val == exp_val)
        per_symptom[sym] = {
            "expected": exp_val,
            "got":      got_val,
            "correct":  is_correct,
            "skip":     False
        }
        evaluated += 1
        if is_correct:
            correct += 1

    case_accuracy = (correct / evaluated) if evaluated > 0 else None

    # Generamos un string amigable para la "Respuesta del Modelo"
    model_response_str = ", ".join([f"{s}: {'Si' if v.get('positive') else 'No'}" for s, v in extracted.items()]) or "Nada detectado"

    return {
        "id":            case["id"],
        "question":      case["question"],
        "text":          case["text"],
        "extracted":     extracted,
        "model_response": model_response_str,
        "per_symptom":   per_symptom,
        "case_accuracy": case_accuracy,
        "evaluated":     evaluated,
        "correct":       correct,
    }


def compute_metrics(results):
    """
    Calcula precision, recall y F1 por síntoma y en global.
    """
    stats = {sym: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for sym in SYMPTOM_KEYS}

    for r in results:
        for sym in SYMPTOM_KEYS:
            info = r["per_symptom"][sym]
            if info["skip"]:
                continue
            exp = info["expected"]
            got = info["got"]

            if exp is True and got is True:
                stats[sym]["TP"] += 1
            elif exp is True and got is not True:   # False o None
                stats[sym]["FN"] += 1
            elif exp is False and got is False:
                stats[sym]["TN"] += 1
            elif exp is False and got is not False:  # True o None
                stats[sym]["FP"] += 1

    metrics = {}
    for sym, s in stats.items():
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics[sym] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": s["TN"],
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
        }

    # Macro-average
    avg_p  = sum(m["precision"] for m in metrics.values()) / len(metrics)
    avg_r  = sum(m["recall"]    for m in metrics.values()) / len(metrics)
    avg_f1 = sum(m["f1"]        for m in metrics.values()) / len(metrics)
    metrics["__macro__"] = {
        "precision": round(avg_p, 3),
        "recall":    round(avg_r, 3),
        "f1":        round(avg_f1, 3),
    }

    return metrics


def print_report(results, metrics):
    """Imprime el informe completo de evaluacion en consola."""
    sep  = "=" * 70
    sep2 = "-" * 70

    print(f"\n{sep}")
    print("  [EVAL] EVALUACION DEL EXTRACTOR DE SINTOMAS -- DermaScan")
    print(f"{sep}\n")

    # -- Resultados por caso
    print("RESULTADOS POR CASO:")
    print(sep2)

    errors = []
    total_evaluated = 0
    total_correct   = 0

    for r in results:
        acc_str = f"{r['case_accuracy']*100:.0f}%" if r["case_accuracy"] is not None else "N/A"
        status  = "[OK]" if r["case_accuracy"] == 1.0 else ("[~~ ]" if r["case_accuracy"] and r["case_accuracy"] >= 0.5 else "[ERR]")
        print(f" {status} [{r['id']}] {acc_str:>4}  | Pregunta: {r['question']:10} | \"{r['text'][:50]}{'...' if len(r['text'])>50 else ''}\"")

        if r["case_accuracy"] is not None and r["case_accuracy"] < 1.0:
            for sym, info in r["per_symptom"].items():
                if info["skip"] or info["correct"]:
                    continue
                errors.append({
                    "case_id":  r["id"],
                    "sintoma":  sym,
                    "expected": info["expected"],
                    "got":      info["got"],
                    "text":     r["text"],
                })

        if r["evaluated"] > 0:
            total_evaluated += r["evaluated"]
            total_correct   += r["correct"]

    # -- Resumen de errores
    if errors:
        print(f"\n{sep2}")
        print("DETALLE DE ERRORES:")
        print(sep2)
        for e in errors:
            exp_str = "Si" if e["expected"] else "No"
            got_str = "Si" if e["got"] is True else ("No" if e["got"] is False else "No detectado (None)")
            print(f"  [!] [{e['case_id']}] Sintoma '{e['sintoma']}': esperado={exp_str}, obtenido={got_str}")
            print(f"      Texto: \"{e['text']}\"")
    else:
        print("\n  [OK] Sin errores en ningun caso!")

    # -- Metricas por sintoma
    print(f"\n{sep2}")
    print("METRICAS POR SINTOMA (Precision / Recall / F1):")
    print(sep2)
    print(f"  {'Sintoma':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
    for sym in SYMPTOM_KEYS:
        m = metrics[sym]
        p_bar = "#" * int(m["precision"] * 10)
        print(f"  {sym:<12} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
              f"{m['TP']:>4} {m['FP']:>4} {m['FN']:>4} {m['TN']:>4}  {p_bar}")

    # -- Resumen global
    macro = metrics["__macro__"]
    global_acc = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    print(f"\n{sep}")
    print("RESUMEN GLOBAL:")
    print(f"  Casos evaluados        : {len(results)}")
    print(f"  Decisiones evaluadas   : {total_evaluated}")
    print(f"  Decisiones correctas   : {total_correct}")
    print(f"  Exactitud global       : {global_acc:.1f}%")
    print(f"  Macro-Precision        : {macro['precision']:.3f}")
    print(f"  Macro-Recall           : {macro['recall']:.3f}")
    print(f"  Macro-F1               : {macro['f1']:.3f}")
    print(f"{sep}\n")


def save_results_json(results, metrics, output_path):
    """Guarda el informe completo en JSON."""
    report = {
        "total_cases":     len(results),
        "metrics":         metrics,
        "cases":           results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Ejecutando evaluación del extractor de síntomas...")

    results = [evaluate_case(case) for case in EVALUATION_DATASET]
    metrics = compute_metrics(results)
    print_report(results, metrics)

    # Guardar JSON en la misma carpeta evaluation/
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "evaluation_results.json")
    save_results_json(results, metrics, out_path)
    print(f"  Informe guardado en: {out_path}")
