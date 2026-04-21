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
        # Using lbfgs as it is more robust for multiclass/multioutput
        self.classifier = MultiOutputClassifier(LogisticRegression(C=10, solver='lbfgs', max_iter=1000))
        self.is_trained = False
        # Updated symptoms to English keys to match training_data.py
        self.symptoms = ["pain", "itching", "size", "bleeding", "color"]

    def train(self):
        """Trains the model using data from training_data.py."""
        texts = [case["text"] for case in TRAINING_DATA]
        
        # Prepare labels
        y = []
        for case in TRAINING_DATA:
            row = [case["labels"].get(s, 0) for s in self.symptoms]
            y.append(row)
        
        y = np.array(y)
        
        # Transform texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train
        self.classifier.fit(X, y)
        self.is_trained = True
        print(f"Symptom model trained with {len(texts)} examples.")
        self.save()

    def save(self):
        """Saves the model and vectorizer to a pkl file."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved in {self.model_path}")

    def load(self):
        """Loads the model from the pkl file."""
        if not os.path.exists(self.model_path):
            print("No trained model found. Falling back to Regex.")
            return False
            
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier']
                self.is_trained = data['is_trained']
            print("Symptom model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, text):
        """Predicts symptoms for a given text."""
        if not self.is_trained:
            return None
            
        X = self.vectorizer.transform([text.lower()])
        prediction = self.classifier.predict(X)[0]
        
        results = {}
        for i, symptom in enumerate(self.symptoms):
            results[symptom] = bool(prediction[i])
            
        return results
