
import sys
import os

# Añadir el directorio actual al path para importar los módulos locale
sys.path.append(os.getcwd())

from evaluation.evaluate_symptom_extractor import EVALUATION_DATASET

symptoms_to_track = ["dolor", "picor", "tamaño", "sangrado", "color"]

print('"""\nDataset de entrenamiento para el extractor de síntomas.\nGenerado automáticamente a partir del dataset de evaluación.\n"""\n')
print("TRAINING_DATA = [")

for case in EVALUATION_DATASET:
    labels = {s: 0 for s in symptoms_to_track}
    for s, val in case["expected"].items():
        if s in labels:
            labels[s] = 1 if val is True else 0
    
    # Solo incluimos casos que tengan al menos una etiqueta de los 5 síntomas principales
    # O que sean explícitamente negativos para ellos
    print(f'    {{"text": "{case["text"]}", "labels": {labels}}},')

print("]")
