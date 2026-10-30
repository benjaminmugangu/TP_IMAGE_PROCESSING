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
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\TP7_CNN_CLASSIFICATION_ULPGL")
DATA_DIR = BASE_DIR / "dataset_split_fast"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
IMG_SIZE = 224  # ResNet-18 attend du 224x224
BATCH_SIZE = 16
NUM_EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositif de calcul : {device}")

# -------------------------------------------------------------------------
# 1. Pipelines de données (avec augmentation pour le train)
# -------------------------------------------------------------------------
tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=DATA_DIR / "train", transform=tf_train)
val_dataset = datasets.ImageFolder(root=DATA_DIR / "val", transform=tf_eval)
test_dataset = datasets.ImageFolder(root=DATA_DIR / "test", transform=tf_eval)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = list(train_dataset.class_to_idx.keys())
print(f"Classes : {CLASS_NAMES}")
print(f"Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# -------------------------------------------------------------------------
# 2. Boucle d'entraînement générique
# -------------------------------------------------------------------------
def train_model(model, optimizer, experiment_name, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT : {experiment_name}")
    print(f"{'='*60}")
    
    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total * 100.0
        
        # --- Val ---
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total * 100.0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if epoch % 3 == 0 or epoch == 1:
            print(f"  Époque [{epoch:2d}/{num_epochs}] Train Loss={train_loss:.4f} (Acc={train_acc:.1f}%) | Val Loss={val_loss:.4f} (Acc={val_acc:.1f}%)")
    
    total_time = time.time() - start_time
    
    # --- Test final ---
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total * 100.0
    
    print(f"  => Temps total : {total_time:.1f}s | Score TEST : {test_acc:.2f}%")
    return history, total_time, test_acc

# -------------------------------------------------------------------------
# 3. STRATÉGIE 1 : Feature Extraction (couches gelées)
# -------------------------------------------------------------------------
print("\n[1/2] Préparation de ResNet-18 en EXTRACTION DE CARACTÉRISTIQUES...")
model_fe = models.resnet18(weights='IMAGENET1K_V1')

# Geler TOUS les paramètres
for param in model_fe.parameters():
    param.requires_grad = False

# Remplacer la tête par une nouvelle couche pour 3 classes
num_features = model_fe.fc.in_features  # 512
model_fe.fc = nn.Linear(num_features, NUM_CLASSES)
# Seule model_fe.fc est entraînable
model_fe = model_fe.to(device)

trainable_params_fe = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
total_params_fe = sum(p.numel() for p in model_fe.parameters())
print(f"  Paramètres entraînables : {trainable_params_fe:,} / {total_params_fe:,} (seulement la tête linéaire)")

optimizer_fe = optim.Adam(model_fe.fc.parameters(), lr=0.001)
hist_fe, time_fe, test_acc_fe = train_model(model_fe, optimizer_fe, "ResNet-18 Feature Extraction (gelé)")

# -------------------------------------------------------------------------
# 4. STRATÉGIE 2 : Fine-Tuning (tout le réseau, lr=1e-5)
# -------------------------------------------------------------------------
print("\n[2/2] Préparation de ResNet-18 en FINE-TUNING...")
model_ft = models.resnet18(weights='IMAGENET1K_V1')

# Remplacer la tête
model_ft.fc = nn.Linear(model_ft.fc.in_features, NUM_CLASSES)
model_ft = model_ft.to(device)
# Tous les paramètres sont dégelés par défaut

trainable_params_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
print(f"  Paramètres entraînables : {trainable_params_ft:,} / {trainable_params_ft:,} (réseau ENTIER)")

optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-5)  # lr très petit !
hist_ft, time_ft, test_acc_ft = train_model(model_ft, optimizer_ft, "ResNet-18 Fine-Tuning (lr=1e-5)")

# -------------------------------------------------------------------------
# 5. Courbes comparatives Extraction vs Fine-Tuning
# -------------------------------------------------------------------------
print("\n[+] Tracé des courbes comparatives ResNet-18...")
epochs_range = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(epochs_range, hist_fe['train_loss'], 'r--', label='Train (Extraction)', alpha=0.8)
ax1.plot(epochs_range, hist_fe['val_loss'], 'r-', label='Val (Extraction)', linewidth=2)
ax1.plot(epochs_range, hist_ft['train_loss'], 'm--', label='Train (Fine-Tune)', alpha=0.8)
ax1.plot(epochs_range, hist_ft['val_loss'], 'm-', label='Val (Fine-Tune)', linewidth=2)
ax1.set_title("Loss : Extraction vs Fine-Tuning", fontsize=12, fontweight='bold')
ax1.set_xlabel("Époques"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.6)

ax2.plot(epochs_range, hist_fe['train_acc'], 'r--', label='Train (Extraction)', alpha=0.8)
ax2.plot(epochs_range, hist_fe['val_acc'], 'r-', label='Val (Extraction)', linewidth=2)
ax2.plot(epochs_range, hist_ft['train_acc'], 'm--', label='Train (Fine-Tune)', alpha=0.8)
ax2.plot(epochs_range, hist_ft['val_acc'], 'm-', label='Val (Fine-Tune)', linewidth=2)
ax2.set_title("Précision : Extraction vs Fine-Tuning", fontsize=12, fontweight='bold')
ax2.set_xlabel("Époques"); ax2.set_ylabel("Précision (%)")
ax2.legend(); ax2.grid(True, linestyle=":", alpha=0.6)

plt.suptitle("TP7 — ResNet-18 : Feature Extraction vs Fine-Tuning", fontsize=14, fontweight='bold')
fig_path = OUTPUT_DIR / "courbes_resnet18_extraction_vs_finetuning.png"
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> Courbes : {fig_path}")

# -------------------------------------------------------------------------
# 6. Sauvegarde des métriques pour le tableau comparatif final
# -------------------------------------------------------------------------
results = {
    'resnet18_feature_extraction': {
        'val_acc': hist_fe['val_acc'][-1],
        'test_acc': test_acc_fe,
        'train_time_sec': time_fe,
        'trainable_params': trainable_params_fe
    },
    'resnet18_fine_tuning': {
        'val_acc': hist_ft['val_acc'][-1],
        'test_acc': test_acc_ft,
        'train_time_sec': time_ft,
        'trainable_params': trainable_params_ft
    }
}
with open(OUTPUT_DIR / "metrics_resnet18.json", "w") as f:
    json.dump(results, f, indent=4)
print("  -> Métriques ResNet-18 enregistrées dans metrics_resnet18.json")
