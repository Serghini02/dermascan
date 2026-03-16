"""
CNN para clasificación de lesiones cutáneas — ResNet18 fine-tuned.
Entrenada con HAM10000 (10015 imágenes, 7 clases).
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_CONFIG, DIAGNOSIS_CLASSES, RISK_LEVELS


class SkinLesionCNN(nn.Module):
    """ResNet18 fine-tuned para clasificación de lesiones cutáneas."""

    def __init__(self, num_classes=7):
        super().__init__()
        # ResNet18 pre-entrenada
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Congelar las primeras capas (transfer learning)
        for param in list(self.base_model.parameters())[:-20]:
            param.requires_grad = False

        # Reemplazar última capa fully connected
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


# Transformaciones para datos de entrada
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((CNN_CONFIG["image_size"], CNN_CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((CNN_CONFIG["image_size"], CNN_CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


class SkinClassifier:
    """Clasificador de lesiones cutáneas con CNN pre-entrenada."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SkinLesionCNN(CNN_CONFIG["num_classes"]).to(self.device)
        self.model.eval()
        self.loaded = False

    def load_model(self, path=None):
        """Carga modelo entrenado."""
        path = path or CNN_CONFIG["model_path"]
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.loaded = True
            print(f"[CNN] Modelo cargado desde {path}")
            return True
        print(f"[CNN] No se encontró modelo en {path}")
        return False

    def predict(self, image):
        """
        Clasifica una imagen de lesión cutánea.

        Args:
            image: PIL Image o numpy array (BGR)

        Returns:
            dict con diagnosis, probabilities, risk_level, confidence
        """
        # Convertir si es numpy array (BGR -> RGB PIL)
        if isinstance(image, np.ndarray):
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Preprocesar
        transform = data_transforms["val"]
        img_tensor = transform(image).unsqueeze(0).to(self.device)

        # Predecir
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Resultado
        predicted_idx = int(np.argmax(probs))
        diagnosis_info = DIAGNOSIS_CLASSES[predicted_idx]
        risk_info = RISK_LEVELS[diagnosis_info["risk"]]

        # Probabilidades por clase
        probabilities = {}
        for idx, prob in enumerate(probs):
            class_info = DIAGNOSIS_CLASSES[idx]
            probabilities[class_info["code"]] = {
                "name": class_info["name"],
                "probability": round(float(prob), 4),
                "risk": class_info["risk"],
            }

        return {
            "diagnosis_code": diagnosis_info["code"],
            "diagnosis_name": diagnosis_info["name"],
            "confidence": round(float(probs[predicted_idx]), 4),
            "risk_level": diagnosis_info["risk"],
            "risk_label": risk_info["label"],
            "risk_color": risk_info["color"],
            "probabilities": probabilities,
            "prob_vector": [round(float(p), 4) for p in probs],
        }

    def predict_demo(self):
        """Predicción demo sin imagen (con probabilidades simuladas)."""
        # Simular distribución realista
        probs = np.random.dirichlet(np.ones(7) * 0.5)
        predicted_idx = int(np.argmax(probs))
        diagnosis_info = DIAGNOSIS_CLASSES[predicted_idx]
        risk_info = RISK_LEVELS[diagnosis_info["risk"]]

        probabilities = {}
        for idx, prob in enumerate(probs):
            class_info = DIAGNOSIS_CLASSES[idx]
            probabilities[class_info["code"]] = {
                "name": class_info["name"],
                "probability": round(float(prob), 4),
                "risk": class_info["risk"],
            }

        return {
            "diagnosis_code": diagnosis_info["code"],
            "diagnosis_name": diagnosis_info["name"],
            "confidence": round(float(probs[predicted_idx]), 4),
            "risk_level": diagnosis_info["risk"],
            "risk_label": risk_info["label"],
            "risk_color": risk_info["color"],
            "probabilities": probabilities,
            "prob_vector": [round(float(p), 4) for p in probs],
            "demo_mode": True,
        }
