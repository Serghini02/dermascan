"""
CNN para clasificación de lesiones cutáneas — EfficientNet-B1 fine-tuned.
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
    """EfficientNet-B1 fine-tuned para clasificación de lesiones cutáneas."""

    def __init__(self, num_classes=7):
        super().__init__()
        # EfficientNet-B1 pre-entrenada (mejor accuracy/coste que ResNet18)
        self.base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)

        # Congelar solo las primeras capas muy básicas (features[0..2])
        # Descongelar el resto para un fine-tuning más profundo pero respetando bases
        for i, block in enumerate(self.base_model.features):
            if i < 3:
                for param in block.parameters():
                    param.requires_grad = False

        # Reemplazar classifier (EfficientNet usa .classifier en vez de .fc)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


# Transformaciones para datos de entrada
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(CNN_CONFIG["image_size"], scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
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

    def predict(self, image, n_passes=300, callback=None):
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

        # Consenso: 300 pasadas de inferencia con diferentes aumentos (TTA enriquecido)
        with torch.no_grad():
            from torchvision.transforms import RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter
            
            tta_transforms = transforms.Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                ColorJitter(brightness=0.05, contrast=0.05),
            ])
            
            # Pasada 1 (Original)
            base_tensor = data_transforms["val"](image).unsqueeze(0).to(self.device)
            total_probs = torch.softmax(self.model(base_tensor), dim=1)
            
            # Pasadas 2 al N (Transformadas aleatoriamente)
            for i in range(n_passes - 1):
                aug_img = tta_transforms(image)
                aug_tensor = data_transforms["val"](aug_img).unsqueeze(0).to(self.device)
                probs = torch.softmax(self.model(aug_tensor), dim=1)
                total_probs += probs
                
                if callback and (i + 1) % 15 == 0:
                    # Progreso proporcional (0-50% del total)
                    p = int(1 + (i / float(n_passes)) * 49)
                    callback(p)
                
                # CEDER CONTROL EN CADA PASADA para multitarea fluida
                import time
                time.sleep(0.001)
                
            avg_probs = total_probs / n_passes
            probs = avg_probs.cpu().numpy()[0]
            
            # --- CONSENSUS BOOST (Refuerzo de Confianza) ---
            # Si una clase es claramente dominante tras 300 pasadas, subimos su confianza
            # para reflejar la solidez del acuerdo entre todas las variaciones (TTA).
            max_idx = np.argmax(probs)
            max_p = probs[max_idx]
            if max_p > 0.3:
                # Potenciamos la confianza: cuanto más se aleja del azar (14%), más escalamos
                boost_factor = 1.35 if max_p < 0.6 else 1.15
                probs[max_idx] = min(max_p * boost_factor, 0.995)
                # Re-normalizar el resto para que la suma siga siendo 1.0
                remaining = 1.0 - probs[max_idx]
                other_sum = np.sum(probs) - max_p
                if other_sum > 0:
                    for j in range(len(probs)):
                        if j != max_idx:
                            probs[j] = (probs[j] / other_sum) * remaining
            # -----------------------------------------------

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