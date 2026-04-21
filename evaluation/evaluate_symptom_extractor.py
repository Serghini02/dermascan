"""
=============================================================================
SYMPTOM EXTRACTOR EVALUATION -- DermaScan
=============================================================================
This script builds a test dataset with real patient phrases,
passes them through `extract_symptoms()` and measures system accuracy.

Calculated metrics per symptom and global:
  - Precision, Recall, F1-Score
  - Case Accuracy (percentage of correctly extracted symptoms)
  - Detailed error report (false positives / negatives)

Usage:
    python evaluation/evaluate_symptom_extractor.py
=============================================================================
"""

import sys
import os
import json

# Force UTF-8 on stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure nlp module is importable from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.symptom_extractor import extract_symptoms

# =============================================================================
# EVALUATION DATASET (English Version)
# =============================================================================
EVALUATION_DATASET = [
    # --- PAIN ---
    { "id": "D-01", "question": "pain", "text": "Yes, it hurts quite a bit when I touch it.", "expected": {"pain": True} },
    { "id": "D-02", "question": "pain", "text": "It doesn't hurt at all, I feel nothing.", "expected": {"pain": False} },
    { "id": "D-03", "question": "pain", "text": "I have some discomfort when I press the area.", "expected": {"pain": True} },
    { "id": "D-04", "question": "pain", "text": "Sometimes it burns a little, especially at night.", "expected": {"pain": True} },
    { "id": "D-05", "question": "pain", "text": "I don't feel anything special, it's there but without pain.", "expected": {"pain": False} },
    { "id": "D-06", "question": "pain", "text": "Not at all, there is no type of pain.", "expected": {"pain": False} },
    { "id": "D-07", "question": "pain", "text": "It stings a bit if I wear tight clothes.", "expected": {"pain": True} },
    { "id": "D-08", "question": "pain", "text": "I get sharp stabs from time to time.", "expected": {"pain": True} },
    { "id": "D-09", "question": "pain", "text": "I don't experience any discomfort when pressing.", "expected": {"pain": False} },
    { "id": "D-10", "question": "pain", "text": "It's a dull but constant pain.", "expected": {"pain": True} },
    { "id": "D-12", "question": "pain", "text": "Fortunately it doesn't hurt at all.", "expected": {"pain": False} },
    { "id": "D-13", "question": "pain", "text": "It's very sensitive to touch, it hurts.", "expected": {"pain": True} },
    { "id": "D-16", "question": "pain", "text": "No pain at all, zero discomfort.", "expected": {"pain": False} },

    # --- ITCHING ---
    { "id": "P-01", "question": "itching", "text": "Yes, it itches a lot, especially in the afternoon.", "expected": {"itching": True} },
    { "id": "P-02", "question": "itching", "text": "It doesn't itch, I've never noticed itching.", "expected": {"itching": False} },
    { "id": "P-03", "question": "itching", "text": "I have itching from time to time, not always.", "expected": {"itching": True} },
    { "id": "P-04", "question": "itching", "text": "Sometimes it itches a bit, yes.", "expected": {"itching": True} },
    { "id": "P-05", "question": "itching", "text": "No itching, everything normal in that area.", "expected": {"itching": False} },
    { "id": "P-06", "question": "itching", "text": "Truth is I never scratch, it doesn't itch.", "expected": {"itching": False} },
    { "id": "P-07", "question": "itching", "text": "I feel an overwhelming need to scratch.", "expected": {"itching": True} },
    { "id": "P-08", "question": "itching", "text": "I don't have itching, the area is very calm.", "expected": {"itching": False} },
    { "id": "P-11", "question": "itching", "text": "I don't notice it itching at all.", "expected": {"itching": False} },
    { "id": "P-14", "question": "itching", "text": "It makes me want to scratch all the time.", "expected": {"itching": True} },
    { "id": "P-15", "question": "itching", "text": "It doesn't itch nor bother me.", "expected": {"itching": False} },

    # --- SIZE ---
    { "id": "T-01", "question": "size", "text": "Yes, it has grown quite a bit in recent months.", "expected": {"size": True} },
    { "id": "T-02", "question": "size", "text": "No, it's still the same size as always.", "expected": {"size": False} },
    { "id": "T-03", "question": "size", "text": "I think it's larger than before.", "expected": {"size": True} },
    { "id": "T-04", "question": "size", "text": "I don't think it has changed, I see it the same as always.", "expected": {"size": False} },
    { "id": "T-05", "question": "size", "text": "It has increased in size notably this year.", "expected": {"size": True} },
    { "id": "T-07", "question": "size", "text": "It's identical to when I got it, it hasn't varied at all.", "expected": {"size": False} },
    { "id": "T-08", "question": "size", "text": "It has become much bulkier and larger.", "expected": {"size": True} },
    { "id": "T-09", "question": "size", "text": "It remains exactly the same as ten years ago.", "expected": {"size": False} },
    { "id": "T-11", "question": "size", "text": "It has doubled its size in just a month.", "expected": {"size": True} },
    { "id": "T-12", "question": "size", "text": "It hasn't grown even a millimeter, it's the same.", "expected": {"size": False} },
    { "id": "T-14", "question": "size", "text": "Its size is constant, it hasn't changed at all.", "expected": {"size": False} },

    # --- BLEEDING ---
    { "id": "S-01", "question": "bleeding", "text": "Yes, once it bled when I scratched it.", "expected": {"bleeding": True} },
    { "id": "S-02", "question": "bleeding", "text": "No, it has never bled at any time.", "expected": {"bleeding": False} },
    { "id": "S-03", "question": "bleeding", "text": "Sometimes it bleeds a little bit, for no apparent reason.", "expected": {"bleeding": True} },
    { "id": "S-04", "question": "bleeding", "text": "No bleeding at all, it's completely dry.", "expected": {"bleeding": False} },
    { "id": "S-06", "question": "bleeding", "text": "It has never released blood or anything like that.", "expected": {"bleeding": False} },
    { "id": "S-07", "question": "bleeding", "text": "I woke up with the sheet stained with blood.", "expected": {"bleeding": True} },
    { "id": "S-08", "question": "bleeding", "text": "It doesn't bleed even if it gets caught in clothes.", "expected": {"bleeding": False} },
    { "id": "S-10", "question": "bleeding", "text": "It has formed a blood scab.", "expected": {"bleeding": True} },
    { "id": "S-11", "question": "bleeding", "text": "No episodes of bleeding to date.", "expected": {"bleeding": False} },
    { "id": "S-13", "question": "bleeding", "text": "It's very dry, it never bleeds.", "expected": {"bleeding": False} },
    { "id": "S-15", "question": "bleeding", "text": "It hasn't bled even once.", "expected": {"bleeding": False} },

    # --- COLOR ---
    { "id": "C-01", "question": "color", "text": "Yes, it has darkened quite a bit, it used to be lighter.", "expected": {"color": True} },
    { "id": "C-02", "question": "color", "text": "It hasn't changed color, it's the same.", "expected": {"color": False} },
    { "id": "C-03", "question": "color", "text": "It has different colors, like spots inside the mole.", "expected": {"color": True} },
    { "id": "C-04", "question": "color", "text": "The color is the same as always, uniform and brown.", "expected": {"color": False} },
    { "id": "C-05", "question": "color", "text": "It changed color, now it has reddish parts.", "expected": {"color": True} },
    { "id": "C-07", "question": "color", "text": "I don't notice any chromatic variation, it's the same.", "expected": {"color": False} },
    { "id": "C-09", "question": "color", "text": "It maintains its uniform brown tone as always.", "expected": {"color": False} },
    { "id": "C-12", "question": "color", "text": "There has been no change in pigmentation.", "expected": {"color": False} },
    { "id": "C-14", "question": "color", "text": "The color is the same, it hasn't mutated.", "expected": {"color": False} },

    # --- DURATION ---
    { "id": "DU-01", "question": "duration", "text": "I've had it for about 3 months.", "expected": {"duration": True} },
    { "id": "DU-02", "question": "duration", "text": "I started noticing it about 2 years ago.", "expected": {"duration": True} },
    { "id": "DU-03", "question": "duration", "text": "I've had it all my life, since I was born.", "expected": {"duration": True} },
    { "id": "DU-04", "question": "duration", "text": "Recently noticed changes, a few weeks ago.", "expected": {"duration": True} },
    { "id": "DU-06", "question": "duration", "text": "It appeared just a few days ago.", "expected": {"duration": True} },
    { "id": "DU-08", "question": "duration", "text": "I've been with this for more than a decade.", "expected": {"duration": True} },
    { "id": "DU-12", "question": "duration", "text": "I discovered it only three weeks ago.", "expected": {"duration": True} },

    # --- MIXED ---
    { "id": "MX-01", "question": "general", "text": "It itches and has also changed color, it's darker.", "expected": {"itching": True, "color": True} },
    { "id": "MX-02", "question": "general", "text": "It hurts a little and bled the other day.", "expected": {"pain": True, "bleeding": True} },
    { "id": "MX-03", "question": "general", "text": "It doesn't hurt, it doesn't itch and it hasn't changed size.", "expected": {"pain": False, "itching": False, "size": False} },

    # --- SHORT ---
    { "id": "SC-01", "question": "pain", "text": "Yes.", "expected": {"pain": True} },
    { "id": "SC-02", "question": "pain", "text": "No.", "expected": {"pain": False} },
    { "id": "SC-03", "question": "itching", "text": "A little bit.", "expected": {"itching": True} },
    { "id": "SC-04", "question": "bleeding", "text": "Never.", "expected": {"bleeding": False} },
]

# =============================================================================
# EVALUATION LOGIC
# =============================================================================

SYMPTOM_KEYS = ["pain", "itching", "size", "bleeding", "color", "duration"]

def evaluate_case(case):
    ctx = case["question"] if case["question"] in SYMPTOM_KEYS else None
    extracted = extract_symptoms(case["text"], context_symptom=ctx)
    expected = case["expected"]

    per_symptom = {}
    evaluated = 0
    correct = 0

    for sym in SYMPTOM_KEYS:
        exp_val = expected.get(sym)
        got_val = extracted.get(sym, {}).get("positive")

        if exp_val is None:
            per_symptom[sym] = {"expected": None, "got": got_val, "correct": None, "skip": True}
            continue

        is_correct = (got_val == exp_val)
        per_symptom[sym] = {
            "expected": exp_val,
            "got": got_val,
            "correct": is_correct,
            "skip": False
        }
        evaluated += 1
        if is_correct:
            correct += 1

    case_accuracy = (correct / evaluated) if evaluated > 0 else 1.0

    model_response_str = ", ".join([f"{s}: {'Yes' if v.get('positive') else 'No'}" for s, v in extracted.items()])
    return {
        "id": case["id"],
        "question": case["question"],
        "text": case["text"],
        "model_response": model_response_str,
        "per_symptom": per_symptom,
        "case_accuracy": case_accuracy,
        "evaluated": evaluated,
        "correct": correct,
    }

def compute_metrics(results):
    stats = {sym: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for sym in SYMPTOM_KEYS}
    total_evaluated = 0
    total_correct = 0

    for r in results:
        total_evaluated += r["evaluated"]
        total_correct += r["correct"]
        for sym in SYMPTOM_KEYS:
            info = r["per_symptom"][sym]
            if info["skip"]: continue
            exp, got = info["expected"], info["got"]
            if exp is True and got is True: stats[sym]["TP"] += 1
            elif exp is True: stats[sym]["FN"] += 1
            elif exp is False and got is False: stats[sym]["TN"] += 1
            elif exp is False: stats[sym]["FP"] += 1

    metrics = {}
    for sym, s in stats.items():
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 1.0
        metrics[sym] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "TP": tp, "FP": fp, "FN": fn, "TN": s["TN"]}

    global_acc = total_correct / total_evaluated if total_evaluated > 0 else 1.0
    return metrics, global_acc

if __name__ == "__main__":
    results = [evaluate_case(c) for c in EVALUATION_DATASET]
    metrics, global_acc = compute_metrics(results)
    
    report = {
        "total_cases": len(results),
        "global_accuracy": global_acc,
        "metrics": metrics,
        "cases": results,
    }
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Evaluation complete. Global Accuracy: {global_acc*100:.2f}%")
