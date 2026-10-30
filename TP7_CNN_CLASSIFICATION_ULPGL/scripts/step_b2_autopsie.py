import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# -------------------------------------------------------------------------
# 1. Configuration et Chemins
# -------------------------------------------------------------------------
BASE_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\TP7_CNN_CLASSIFICATION_ULPGL")
DATA_DIR = BASE_DIR / "dataset_split_fast"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
IMG_SIZE = 224
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=DATA_DIR / "val", transform=tf_eval)
test_dataset = datasets.ImageFolder(root=DATA_DIR / "test", transform=tf_eval)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = list(val_dataset.class_to_idx.keys())
print(f"Classes détectées : {CLASS_NAMES}")

# -------------------------------------------------------------------------
# 2. Ré-entraînement rapide du modèle Fine-Tuning optimal pour extraction d'erreurs
# -------------------------------------------------------------------------
print("\n[+] Chargement et inférence du modèle ResNet-18 Fine-Tuning...")
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Entraînement ciblé 10 époques avec lr=1e-5 (reproductible seed 42)
torch.manual_seed(42)
tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(root=DATA_DIR / "train", transform=tf_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(1, 11):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

# -------------------------------------------------------------------------
# 3. Évaluation sur Validation et Test + Matrices de Confusion
# -------------------------------------------------------------------------
def evaluate_dataset(loader, dataset_name):
    model.eval()
    all_preds, all_labels, all_probs, all_paths = [], [], [], []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Récupérer les chemins
    all_paths = [s[0] for s in loader.dataset.samples]
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_paths

val_preds, val_labels, val_probs, val_paths = evaluate_dataset(val_loader, "Validation")
test_preds, test_labels, test_probs, test_paths = evaluate_dataset(test_loader, "Test")

cm_val = confusion_matrix(val_labels, val_preds)
cm_test = confusion_matrix(test_labels, test_preds)

print("\n=== MATRICE DE CONFUSION SUR VALIDATION ===")
print(cm_val)
print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES))

print("\n=== MATRICE DE CONFUSION SUR TEST ===")
print(cm_test)
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

# Tracé des deux matrices de confusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1, cbar=False)
ax1.set_title("Matrice de Confusion — Validation", fontsize=12, fontweight='bold')
ax1.set_xlabel("Classe Prédite"); ax1.set_ylabel("Vraie Classe")

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2, cbar=False)
ax2.set_title("Matrice de Confusion — Test", fontsize=12, fontweight='bold')
ax2.set_xlabel("Classe Prédite"); ax2.set_ylabel("Vraie Classe")

plt.suptitle("TP7 — ResNet-18 Fine-Tuning : Matrices de Confusion (Goma Currency Dataset)", fontsize=13, fontweight='bold')
cm_path = OUTPUT_DIR / "matrice_confusion_resnet18_val_test.png"
plt.savefig(cm_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> Matrices de confusion sauvegardées : {cm_path}")

# -------------------------------------------------------------------------
# 4. Planche des Erreurs et Cas d'Incertitude (Partie B.2 du sujet)
# -------------------------------------------------------------------------
# Trouver les erreurs réelles
errors_idx = np.where(val_preds != val_labels)[0]
print(f"\n[+] Nombre d'erreurs en Validation : {len(errors_idx)} sur {len(val_labels)}")

# Pour avoir les pires cas (soit erreurs, soit les plus faibles probabilités de la vraie classe)
# Mesure de perte de confiance : 1 - probabilité attribuée à la vraie classe
uncertainty = [1.0 - val_probs[i][val_labels[i]] for i in range(len(val_labels))]
worst_idx = np.argsort(uncertainty)[::-1][:12]  # Les 12 pires cas

print(f"[+] Sélection des 12 pires cas (erreurs et incertitudes maximales)...")

fig, axes = plt.subplots(3, 4, figsize=(15, 11))
plt.subplots_adjust(wspace=0.3, hspace=0.45)

for i, idx in enumerate(worst_idx):
    row, col = i // 4, i % 4
    ax = axes[row, col]
    
    img_path = val_paths[idx]
    img_pil = Image.open(img_path).convert("RGB")
    
    true_cls = CLASS_NAMES[val_labels[idx]]
    pred_cls = CLASS_NAMES[val_preds[idx]]
    conf = val_probs[idx][val_preds[idx]] * 100
    
    is_correct = (true_cls == pred_cls)
    title_color = "green" if is_correct else "red"
    status = "OK" if is_correct else "ERREUR"
    
    ax.imshow(img_pil)
    ax.set_title(f"[{status}] Vrai: {true_cls}\nPrédit: {pred_cls} ({conf:.1f}%)", 
                 fontsize=9.5, fontweight='bold', color=title_color)
    ax.axis("off")

plt.suptitle("TP7 — Autopsie : Les 12 Pires Cas d'Incertitude / Erreurs (Partie B.2)", fontsize=14, fontweight='bold', y=0.98)
autopsie_path = OUTPUT_DIR / "autopsie_12_pires_erreurs.png"
plt.savefig(autopsie_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> Planche d'autopsie sauvegardée : {autopsie_path}")

print("\n=== AUTOPSIE ET ANALYSE TECHNIQUE DU SUJET TERMINÉE ===")
