import os
import sys
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Fixer les graines pour la reproductibilité
torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\TP7_CNN_CLASSIFICATION_ULPGL")
DATA_DIR = BASE_DIR / "dataset_split_fast"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
CLASS_NAMES = ["1000_fc", "500_fc", "50_fc"]
IMG_SIZE = 64  # Résolution adaptée pour le petit CNN du cours
BATCH_SIZE = 16
NUM_EPOCHS = 25
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositif de calcul utilisé : {device}")

# -------------------------------------------------------------------------
# 1. Définition de l'Architecture du Petit CNN (Diapositive 19 du cours)
# -------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 64 -> 32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 32 -> 16
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------------------------------------------------------
# 2. Pipelines de Données : Sans vs Avec Augmentation
# -------------------------------------------------------------------------
# Pipeline Standard (Sans Augmentation)
tf_val_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tf_train_no_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pipeline Avec Augmentation de Données
tf_train_with_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=DATA_DIR / "val", transform=tf_val_test)
test_dataset = datasets.ImageFolder(root=DATA_DIR / "test", transform=tf_val_test)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_and_evaluate(train_transform, experiment_name):
    print(f"\n{'='*60}")
    print(f"DÉMARRAGE ENTRAÎNEMENT : {experiment_name.upper()}")
    print(f"{'='*60}")
    
    train_dataset = datasets.ImageFolder(root=DATA_DIR / "train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SmallCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'time_per_epoch': []
    }
    
    start_total_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # --- Mode Entraînement ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0
        
        # --- Mode Validation ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss = val_loss / total_val
        val_acc = (correct_val / total_val) * 100.0
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['time_per_epoch'].append(epoch_time)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Époque [{epoch:2d}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%) | {epoch_time:.2f}s")
            
    total_training_time = time.time() - start_total_time
    print(f"Temps total d'entraînement : {total_training_time:.2f} secondes.")
    
    # Évaluation finale sur le Test
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
    test_acc = (correct_test / total_test) * 100.0
    print(f"Score final sur le TEST : {test_acc:.2f} %")
    
    return history, total_training_time, test_acc, model

# Exécution des deux régimes
hist_no_aug, time_no_aug, test_acc_no_aug, model_no_aug = train_and_evaluate(tf_train_no_aug, "CNN Sans Augmentation")
hist_with_aug, time_with_aug, test_acc_with_aug, model_with_aug = train_and_evaluate(tf_train_with_aug, "CNN Avec Augmentation")

# -------------------------------------------------------------------------
# 3. Tracé des Courbes Comparatives (Partie A.4)
# -------------------------------------------------------------------------
print("\n[+] Génération des courbes comparatives Train/Val...")
epochs = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Courbes de Perte (Loss)
ax1.plot(epochs, hist_no_aug['train_loss'], 'b--', label='Train (Sans Aug)', alpha=0.8)
ax1.plot(epochs, hist_no_aug['val_loss'], 'b-', label='Val (Sans Aug)', linewidth=2)
ax1.plot(epochs, hist_with_aug['train_loss'], 'g--', label='Train (Avec Aug)', alpha=0.8)
ax1.plot(epochs, hist_with_aug['val_loss'], 'g-', label='Val (Avec Aug)', linewidth=2)
ax1.set_title("Évolution de la Perte (Cross-Entropy Loss)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Époques")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

# Courbes de Précision (Accuracy)
ax2.plot(epochs, hist_no_aug['train_acc'], 'b--', label='Train (Sans Aug)', alpha=0.8)
ax2.plot(epochs, hist_no_aug['val_acc'], 'b-', label='Val (Sans Aug)', linewidth=2)
ax2.plot(epochs, hist_with_aug['train_acc'], 'g--', label='Train (Avec Aug)', alpha=0.8)
ax2.plot(epochs, hist_with_aug['val_acc'], 'g-', label='Val (Avec Aug)', linewidth=2)
ax2.set_title("Évolution de la Précision (%)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Époques")
ax2.set_ylabel("Précision (%)")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.suptitle("TP7 — Comparaison du Petit CNN : Sans vs Avec Augmentation de Données", fontsize=14, fontweight='bold')
curve_path = OUTPUT_DIR / "courbes_cnn_avec_sans_augmentation.png"
plt.savefig(curve_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> Courbes sauvegardées : {curve_path}")

# Sauvegarde des métriques au format JSON pour notre tableau comparatif final
results = {
    'cnn_no_aug': {
        'val_acc': hist_no_aug['val_acc'][-1],
        'test_acc': test_acc_no_aug,
        'train_time_sec': time_no_aug
    },
    'cnn_with_aug': {
        'val_acc': hist_with_aug['val_acc'][-1],
        'test_acc': test_acc_with_aug,
        'train_time_sec': time_with_aug
    }
}
with open(OUTPUT_DIR / "metrics_cnn_scratch.json", "w") as f:
    json.dump(results, f, indent=4)
print("  -> Métriques enregistrées dans metrics_cnn_scratch.json")
