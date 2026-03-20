"""
Script de entrenamiento de la CNN para clasificación de lesiones cutáneas.
Usa datos reales de HAM10000.
"""

import os
import sys
import csv
import time
import json
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_CONFIG, CLASS_TO_IDX, MODELS_DIR
from vision.cnn_model import SkinLesionCNN, data_transforms


class HAM10000Dataset(Dataset):
    """Dataset personalizado para imágenes HAM10000."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def find_ham10000_files(dataset_dir):
    """Busca el CSV de metadata y las carpetas de imágenes en el directorio del dataset."""
    csv_path = None
    image_dirs = []

    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            full = os.path.join(root, f)
            if f == "HAM10000_metadata.csv" or f == "HAM10000_metadata":
                csv_path = full
            elif f.endswith(".csv") and "metadata" in f.lower():
                csv_path = csv_path or full

        for d in dirs:
            dirpath = os.path.join(root, d)
            if "ham10000_images" in d.lower() or d.startswith("HAM10000_images"):
                image_dirs.append(dirpath)

    # Si no se encontraron carpetas específicas, buscar en todas las subcarpetas
    if not image_dirs:
        image_dirs = [dataset_dir]

    return csv_path, image_dirs


def load_dataset(dataset_dir):
    """Carga imágenes y labels del HAM10000."""
    csv_path, image_dirs = find_ham10000_files(dataset_dir)

    if not csv_path:
        print(f"[Train CNN] ERROR: No se encontró HAM10000_metadata.csv en {dataset_dir}")
        return [], []

    print(f"[Train CNN] CSV encontrado: {csv_path}")
    print(f"[Train CNN] Carpetas de imágenes: {image_dirs}")

    # Leer metadata
    metadata = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row.get("image_id", "")
            dx = row.get("dx", "")
            if img_id and dx in CLASS_TO_IDX:
                metadata[img_id] = CLASS_TO_IDX[dx]

    print(f"[Train CNN] {len(metadata)} entradas en metadata")

    # Buscar imágenes
    image_paths = []
    labels = []
    found = set()

    for img_dir in image_dirs:
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_id = os.path.splitext(f)[0]
                    if img_id in metadata and img_id not in found:
                        image_paths.append(os.path.join(root, f))
                        labels.append(metadata[img_id])
                        found.add(img_id)

    print(f"[Train CNN] {len(image_paths)} imágenes encontradas")
    print(f"[Train CNN] Distribución de clases: {Counter(labels)}")

    return image_paths, labels


def train_cnn(dataset_dir, epochs=None, callback=None):
    """
    Entrena la CNN con datos HAM10000.

    Args:
        dataset_dir: Ruta al directorio HAM10000
        epochs: Número de épocas
        callback: Función callback para progress updates
    """
    epochs = epochs or CNN_CONFIG["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train CNN] Dispositivo: {device}")

    # Cargar datos
    image_paths, labels = load_dataset(dataset_dir)
    if not image_paths:
        return {"error": "No se encontraron imágenes"}

    # Split train/val (80/20)
    n = len(image_paths)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Datasets
    train_dataset = HAM10000Dataset(train_paths, train_labels, data_transforms["train"])
    val_dataset = HAM10000Dataset(val_paths, val_labels, data_transforms["val"])

    # Weighted sampler para balancear clases
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=CNN_CONFIG["batch_size"],
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CNN_CONFIG["batch_size"],
                            shuffle=False, num_workers=0)

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO CNN — Clasificación de Lesiones Cutáneas")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Epochs: {epochs} | Batch: {CNN_CONFIG['batch_size']}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Modelo, loss con pesos de clase, optimizer con weight_decay
    model = SkinLesionCNN(CNN_CONFIG["num_classes"]).to(device)

    # Calcular pesos de clase inversamente proporcionales a la frecuencia
    all_class_counts = Counter(labels)
    total_samples = sum(all_class_counts.values())
    num_classes = CNN_CONFIG["num_classes"]
    class_weights = torch.tensor(
        [total_samples / (num_classes * all_class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    print(f"[Train CNN] Pesos de clase: {class_weights.cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CNN_CONFIG["learning_rate"],
        weight_decay=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Entrenamiento
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, CNN_CONFIG["model_path"])

        # Callback
        if callback:
            callback(
                epoch=epoch, total_epochs=epochs,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc,
                progress=round((epoch + 1) / epochs * 100, 1),
            )

    results = {
        "total_epochs": epochs,
        "best_val_acc": round(best_val_acc, 2),
        "final_train_acc": round(history["train_acc"][-1], 2),
        "history": history,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "cnn_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Entrenamiento CNN completado. Mejor Val Acc: {best_val_acc:.1f}%\n")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Ruta al directorio HAM10000 (auto-detecta si no se pasa)")
    parser.add_argument("--epochs", type=int, default=CNN_CONFIG["epochs"])
    args = parser.parse_args()

    dataset_dir = args.dataset
    if not dataset_dir:
        # Auto-detectar HAM10000 via kagglehub
        try:
            import kagglehub
            dataset_dir = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
            print(f"[Train CNN] HAM10000 auto-detectado en: {dataset_dir}")
        except Exception as e:
            print(f"[Train CNN] ERROR: No se pudo detectar HAM10000: {e}")
            print("  Usa: python vision/train_cnn.py --dataset <ruta>")
            sys.exit(1)

    train_cnn(dataset_dir, args.epochs)
