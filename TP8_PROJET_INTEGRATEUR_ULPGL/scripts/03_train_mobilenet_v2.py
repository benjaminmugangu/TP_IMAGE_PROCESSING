"""
03_train_mobilenet_v2.py
GIN522 - Vision par Ordinateur & CNN (ULPGL 2025-2026)
Projet Integrateur - Etape Transfer Learning avec MobileNetV2
Auteur: Master 2 Genie Electrique et Informatique
"""

import os
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

def train_mobilenet(
    data_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\dataset_split_224",
    output_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\outputs",
    model_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\models",
    num_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    seed=42
):
    print("=" * 60)
    print("  ETAPE 5 : ENTRAINEMENT CNN MOBILENETV2 (TRANSFER LEARNING)")
    print("=" * 60)

    # 1. Reproductibilite
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Materiel de calcul detecte : {device}")

    # 2. Pipeline de Pretraitement et Data Augmentation
    # Normalisation standard ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    }

    # 3. Chargement des donnees via ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=0
        )
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"[+] Classes ({num_classes}) : {class_names}")
    print(f"[+] Tailles des jeux : Train={dataset_sizes['train']}, Val={dataset_sizes['val']}, Test={dataset_sizes['test']}")

    # 4. Modele MobileNetV2 Pre-entraine
    print("\n[*] Chargement du modele MobileNetV2 pre-entraine sur ImageNet...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Remplacement de la derniere couche (tête de classification)
    # L'entree du classifier de MobileNetV2 est in_features = 1280
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    model = model.to(device)

    # 5. Fonction de Perte et Optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 6. Boucle d'entrainement avec Early Stopping / Checkpointing
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("\n[*] Demarrage de l'apprentissage...")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoque {epoch:02d}/{num_epochs:02d} ---")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f"[{phase.upper():<5}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

            # Checkpointing sur le minimum de validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"  --> Nouveau record de validation ! (Loss={best_val_loss:.4f})")

        print(f"Duree epoque : {time.time() - epoch_start:.1f}s")

    time_elapsed = time.time() - since
    print(f"\n[+] Entrainement termine en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"[+] Meilleur modele atteint a l'Epoque {best_epoch} avec Val Loss = {best_val_loss:.4f}")

    # 7. Sauvegarde du meilleur modele
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_mobilenetv2_billets.pth")
    torch.save(best_model_wts, best_model_path)
    print(f"[+] Poids optimaux sauvegardes dans : {best_model_path}")

    # 8. Trace des Courbes d'Apprentissage
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    # Graphique Perte (Loss)
    plt.subplot(1, 2, 1)
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.axvline(x=best_epoch, color='green', linestyle=':', label=f'Meilleure epoque ({best_epoch})')
    plt.title("Courbes de Perte (Loss) - MobileNetV2", fontsize=11, fontweight='bold')
    plt.xlabel("Epoque")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Graphique Exactitude (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs_range, [acc * 100 for acc in history['val_acc']], 'r--', label='Val Accuracy', linewidth=2)
    plt.axvline(x=best_epoch, color='green', linestyle=':', label=f'Meilleure epoque ({best_epoch})')
    plt.title("Courbes d'Exactitude (Accuracy) - MobileNetV2", fontsize=11, fontweight='bold')
    plt.xlabel("Epoque")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "courbes_apprentissage_cnn.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[+] Graphique des courbes d'apprentissage sauvegarde : {curves_path}")

    # 9. Sauvegarde de l'historique en JSON
    metrics_cnn_path = os.path.join(output_dir, "metrics_mobilenet.json")
    with open(metrics_cnn_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "MobileNetV2",
            "epochs": num_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history
        }, f, indent=2)
    print(f"[+] Historique JSON sauvegarde : {metrics_cnn_path}")

if __name__ == "__main__":
    train_mobilenet()
