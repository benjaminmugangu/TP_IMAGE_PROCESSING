import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# 1. Configuration des chemins et reproductibilité
# -------------------------------------------------------------------------
DATA_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\dataset_split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\output_tp7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["50_fc", "500_fc", "1000_fc"]
CLASS_LABELS = {0: "50 FC", 1: "500 FC", 2: "1000 FC"}

print("=== PARTIE A.2 : CHARGEMENT ET VÉRIFICATION VIA IMAGEFOLDER ===")

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=base_transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=base_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=base_transform)

print(f"Index des classes détecté par ImageFolder : {train_dataset.class_to_idx}")
print(f"Nombre d'échantillons Train : {len(train_dataset)}")
print(f"Nombre d'échantillons Val   : {len(val_dataset)}")
print(f"Nombre d'échantillons Test  : {len(test_dataset)}")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------------------------------------------------------
# 2. Grille de visualisation par classe (Partie A.2)
# -------------------------------------------------------------------------
print("\n[+] Génération de la grille d'inspection visuelle...")
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

for row, cls_name in enumerate(CLASS_NAMES):
    cls_folder = TRAIN_DIR / cls_name
    img_paths = sorted(list(cls_folder.glob("*.jpg")))[:4]
    for col, p in enumerate(img_paths):
        img_bgr = cv2.imread(str(p))
        if img_bgr is not None:
            # Redimensionnement rapide pour affichage fluide
            img_thumb = cv2.resize(img_bgr, (400, 260))
            img_rgb = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB)
            ax = axes[row, col]
            ax.imshow(img_rgb)
            ax.set_title(f"{cls_name} (#{col+1})", fontsize=11, fontweight='bold')
            ax.axis("off")

plt.suptitle("TP7 — Grille de Contrôle des Échantillons Réels de Billets (ULPGL Goma)", fontsize=14, fontweight='bold', y=0.98)
grid_path = OUTPUT_DIR / "grille_verification_billets.png"
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Grille enregistrée avec succès : {grid_path}")

# -------------------------------------------------------------------------
# 3. PARTIE A.3 : LES DEUX RÉFÉRENCES (BASELINES) AVANT LE DEEP LEARNING
# -------------------------------------------------------------------------
print("\n" + "="*60)
print("PARTIE A.3 : DEUX RÉFÉRENCES AVANT LE DEEP LEARNING")
print("="*60)

# Extraction des étiquettes brutes
y_train = [label for _, label in train_dataset.samples]
y_val = [label for _, label in val_dataset.samples]
y_test = [label for _, label in test_dataset.samples]

# 3.1 BASELINE 1 : Classifieur Trivial (Classe Majoritaire)
counts_train = np.bincount(y_train)
majority_class = np.argmax(counts_train)
maj_acc_train = np.mean(np.array(y_train) == majority_class) * 100
maj_acc_val = np.mean(np.array(y_val) == majority_class) * 100
maj_acc_test = np.mean(np.array(y_test) == majority_class) * 100

print(f"\n1. BASELINE TRIVIALE (Classe Majoritaire = {list(train_dataset.class_to_idx.keys())[majority_class]}) :")
print(f"   - Précision Train : {maj_acc_train:.2f} %")
print(f"   - Précision Val   : {maj_acc_val:.2f} %")
print(f"   - Précision Test  : {maj_acc_test:.2f} %")

# 3.2 BASELINE 2 : Histogrammes HSV (Module 1) + k-NN
print("\n2. BASELINE CLASSIQUE : Histogrammes HSV + k-Plus Proches Voisins (k-NN) :")

def extract_hsv_histogram(image_path, bins=(16, 8, 8)):
    """Extrait un histogramme 3D HSV normalisé d'une image redimensionnée."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return np.zeros(bins[0]*bins[1]*bins[2])
    # Redimensionnement rapide pour extraction instantanée
    img_small = cv2.resize(img_bgr, (256, 256))
    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

print("   [+] Extraction rapide des descripteurs HSV sur Train, Val, Test...")
X_train_hsv = np.array([extract_hsv_histogram(p) for p, _ in train_dataset.samples])
X_val_hsv = np.array([extract_hsv_histogram(p) for p, _ in val_dataset.samples])
X_test_hsv = np.array([extract_hsv_histogram(p) for p, _ in test_dataset.samples])

# Test avec k=5
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn.fit(X_train_hsv, y_train)

knn_pred_train = knn.predict(X_train_hsv)
knn_pred_val = knn.predict(X_val_hsv)
knn_pred_test = knn.predict(X_test_hsv)

knn_acc_train = accuracy_score(y_train, knn_pred_train) * 100
knn_acc_val = accuracy_score(y_val, knn_pred_val) * 100
knn_acc_test = accuracy_score(y_test, knn_pred_test) * 100

print(f"\n   Résultats du classifieur k-NN (k={best_k}) :")
print(f"   - Précision Train : {knn_acc_train:.2f} %")
print(f"   - Précision Val   : {knn_acc_val:.2f} %")
print(f"   - Précision Test  : {knn_acc_test:.2f} %")

target_names = [k for k, v in sorted(train_dataset.class_to_idx.items(), key=lambda item: item[1])]
print("\n   Rapport détaillé sur la Validation (k-NN HSV) :")
print(classification_report(y_val, knn_pred_val, target_names=target_names))

print("   Matrice de confusion sur la Validation (k-NN HSV) :")
print(confusion_matrix(y_val, knn_pred_val))
