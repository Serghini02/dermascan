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
    # BLOQUE 1: DOLOR
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "D-01",
        "question": "dolor",
        "text": "Sí, me duele bastante cuando lo toco.",
        "expected": {
            "dolor": True,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "D-02",
        "question": "dolor",
        "text": "No me duele para nada, no siento nada.",
        "expected": {
            "dolor": False,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "D-03",
        "question": "dolor",
        "text": "Tengo algo de molestia cuando presiono la zona.",
        "expected": {
            "dolor": True,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "D-04",
        "question": "dolor",
        "text": "A veces arde un poco, sobre todo por las noches.",
        "expected": {
            "dolor": True,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "D-05",
        "question": "dolor",
        "text": "No siento nada especial, está ahí pero sin dolor.",
        "expected": {
            "dolor": False,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 2: PICOR
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "P-01",
        "question": "picor",
        "text": "Sí, me pica mucho, especialmente por la tarde.",
        "expected": {
            "dolor": None,
            "picor": True,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "P-02",
        "question": "picor",
        "text": "No me pica, nunca he notado picor.",
        "expected": {
            "dolor": None,
            "picor": False,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "P-03",
        "question": "picor",
        "text": "Tengo comezón de vez en cuando, no siempre.",
        "expected": {
            "dolor": None,
            "picor": True,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "P-04",
        "question": "picor",
        "text": "A veces pica un poco, sí.",
        "expected": {
            "dolor": None,
            "picor": True,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "P-05",
        "question": "picor",
        "text": "Sin picor, todo normal en esa zona.",
        "expected": {
            "dolor": None,
            "picor": False,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 3: TAMAÑO
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "T-01",
        "question": "tamaño",
        "text": "Sí, ha crecido bastante en los últimos meses.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": True,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "T-02",
        "question": "tamaño",
        "text": "No, sigue siendo del mismo tamaño que siempre.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": False,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "T-03",
        "question": "tamaño",
        "text": "Me parece que está más grande que antes.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": True,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "T-04",
        "question": "tamaño",
        "text": "Creo que no ha cambiado, lo veo igual que siempre.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": False,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "T-05",
        "question": "tamaño",
        "text": "Ha aumentado de tamaño notablemente este año.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": True,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "T-06",
        "question": "tamaño",
        "text": "Era más pequeño cuando era niño, ahora es más grande.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": True,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 4: SANGRADO
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "S-01",
        "question": "sangrado",
        "text": "Sí, una vez sangró cuando me lo rasqué.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": True,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "S-02",
        "question": "sangrado",
        "text": "No, nunca ha sangrado en ningún momento.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": False,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "S-03",
        "question": "sangrado",
        "text": "A veces sangra un poquito, sin causa aparente.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": True,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "S-04",
        "question": "sangrado",
        "text": "No ningún sangrado, está completamente seco.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": False,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "S-05",
        "question": "sangrado",
        "text": "Ha sangrado dos veces estas semanas.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": True,
            "color": None,
            "duracion": True,   # "semanas" → duración también detectada
        }
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 5: COLOR
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C-01",
        "question": "color",
        "text": "Sí, se ha oscurecido bastante, antes era más claro.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": True,
            "duracion": None,
        }
    },
    {
        "id": "C-02",
        "question": "color",
        "text": "No ha cambiado de color, sigue igual.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": False,
            "duracion": None,
        }
    },
    {
        "id": "C-03",
        "question": "color",
        "text": "Tiene colores diferentes, como manchas dentro del lunar.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": True,
            "duracion": None,
        }
    },
    {
        "id": "C-04",
        "question": "color",
        "text": "El color es el mismo de siempre, uniforme y marrón.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": False,
            "duracion": None,
        }
    },
    {
        "id": "C-05",
        "question": "color",
        "text": "Cambió de color, ahora tiene partes rojizas.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": True,
            "duracion": None,
        }
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 6: DURACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "DU-01",
        "question": "duracion",
        "text": "Lo tengo desde hace unos 3 meses.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": True,
        }
    },
    {
        "id": "DU-02",
        "question": "duracion",
        "text": "Lo empecé a notar hace 2 años aproximadamente.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": True,
        }
    },
    {
        "id": "DU-03",
        "question": "duracion",
        "text": "Lo tengo de toda la vida, desde que nací.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": True,
        }
    },
    {
        "id": "DU-04",
        "question": "duracion",
        "text": "Recientemente noté los cambios, hace unas semanas.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": True,
        }
    },
    {
        "id": "DU-05",
        "question": "duracion",
        "text": "No lo sé exactamente, hace bastante tiempo.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": None,
            "sangrado": None,
            "color": None,
            "duracion": True,
        }
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE 7: CASOS MIXTOS (el paciente menciona varios síntomas a la vez)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "MX-01",
        "question": "general",
        "text": "Me pica y también ha cambiado de color, está más oscuro.",
        "expected": {
            "dolor": None,
            "picor": True,
            "tamaño": None,
            "sangrado": None,
            "color": True,
            "duracion": None,
        }
    },
    {
        "id": "MX-02",
        "question": "general",
        "text": "Duele un poco y sangró el otro día.",
        "expected": {
            "dolor": True,
            "picor": None,
            "tamaño": None,
            "sangrado": True,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "MX-03",
        "question": "general",
        "text": "No duele, no pica y no ha cambiado de tamaño.",
        "expected": {
            "dolor": False,
            "picor": False,
            "tamaño": False,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
    {
        "id": "MX-04",
        "question": "general",
        "text": "Creció bastante y lo noto desde hace 6 semanas.",
        "expected": {
            "dolor": None,
            "picor": None,
            "tamaño": True,
            "sangrado": None,
            "color": None,
            "duracion": True,
        }
    },
    {
        "id": "MX-05",
        "question": "general",
        "text": "No siento dolor ni picor, pero sí noto que ha crecido mucho.",
        "expected": {
            "dolor": False,
            "picor": False,
            "tamaño": True,
            "sangrado": None,
            "color": None,
            "duracion": None,
        }
    },
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
    extracted = extract_symptoms(case["text"])
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

    return {
        "id":            case["id"],
        "question":      case["question"],
        "text":          case["text"],
        "extracted":     extracted,
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
