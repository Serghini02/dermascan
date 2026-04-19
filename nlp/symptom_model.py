import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from .training_data import TRAINING_DATA

class SymptomClassifier:
    def __init__(self, model_path="models/symptom_classifier.pkl"):
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.classifier = MultiOutputClassifier(LogisticRegression(C=10, solver='liblinear'))
        self.is_trained = False
        self.symptoms = ["dolor", "picor", "tamaño", "sangrado", "color"]

    def train(self):
        """Entrena el modelo usando los datos de training_data.py."""
        texts = [case["text"] for case in TRAINING_DATA]
        
        # Preparar etiquetas
        y = []
        for case in TRAINING_DATA:
            row = [case["labels"].get(s, 0) for s in self.symptoms]
            y.append(row)
        
        y = np.array(y)
        
        # Transformar textos
        X = self.vectorizer.fit_transform(texts)
        
        # Entrenar
        self.classifier.fit(X, y)
        self.is_trained = True
        print(f"Modelo de síntomas entrenado con {len(texts)} ejemplos.")
        self.save()

    def save(self):
        """Guarda el modelo y el vectorizador en un archivo pkl."""
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            }, f)
        print(f"Modelo guardado en {self.model_path}")

    def load(self):
        """Carga el modelo desde el archivo pkl."""
        if not os.path.exists(self.model_path):
            print("No se encontró un modelo entrenado. Fallback a Regex.")
            return False
            
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier']
                self.is_trained = data['is_trained']
            print("Modelo de síntomas cargado correctamente.")
            return True
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            return False

    def predict(self, text):
        """Predice los síntomas para un texto dado."""
        if not self.is_trained:
            return None
            
        X = self.vectorizer.transform([text.lower()])
        prediction = self.classifier.predict(X)[0]
        
        results = {}
        for i, symptom in enumerate(self.symptoms):
            results[symptom] = bool(prediction[i])
            
        return results
