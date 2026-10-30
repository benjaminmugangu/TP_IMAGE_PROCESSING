"""
03_train_cnn_transfer.py
GIN522 - Vision par Ordinateur & CNN (ULPGL 2025-2026)
Projet Integrateur - Etape Transfer Learning (ResNet18 / MobileNetV2)
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

def get_model(architecture="resnet18", num_classes=3):
    """
    Instancie un modele pre-entraine avec tete clasiffieur adaptee.
    Supporte ResNet18 (pre-telecharge en cache local) et MobileNetV2.
    """
    if architecture == "resnet18":
        print("[*] Architecture choisie : ResNet18 (charge depuis le cache local)")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif architecture == "mobilenet_v2":
        print("[*] Architecture choisie : MobileNetV2")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Architecture inconnue : {architecture}")
    return model

def train_cnn(
    data_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\dataset_split_224",
    output_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\outputs",
    model_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\models",
    architecture="resnet18",
    num_epochs=15,
    batch_size=16,
    learning_rate=1e-4,
    seed=42
):
    print("=" * 60)
    print(f"  ETAPE 5 : ENTRAINEMENT CNN {architecture.upper()} (TRANSFER LEARNING)")
    print("=" * 60)

    # 1. Reproductibilite stricte
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Materiel de calcul : {device}")

    # 2. Pipeline de Pretraitement et Data Augmentation
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

    # 3. Chargement des donnees
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
    print(f"[+] Echantillons : Train={dataset_sizes['train']}, Val={dataset_sizes['val']}, Test={dataset_sizes['test']}")

    # 4. Instanciation du modele
    model = get_model(architecture=architecture, num_classes=num_classes)
    model = model.to(device)

    # 5. Critere & Optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 6. Boucle d'apprentissage avec Early Stopping
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("\n[*] Lancement des epoques d'apprentissage...")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

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

            # Checkpointing
            if phase == 'val':
                record = ""
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_val_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    record = "  --> [NOUVEAU RECORD VAL LOSS]"
                print(f"Epoque {epoch:02d}/{num_epochs:02d} | Train Loss: {history['train_loss'][-1]:.4f} Acc: {history['train_acc'][-1]*100:.1f}% | Val Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.1f}% ({time.time() - epoch_start:.1f}s){record}")

    total_time = time.time() - since
    print(f"\n[+] Entrainement termine en {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"[+] Meilleur modele sauvegarde a l'Epoque {best_epoch} (Val Loss = {best_val_loss:.4f}, Val Acc = {best_val_acc*100:.2f}%)")

    # 7. Sauvegarde des poids optimaux
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"best_{architecture}_billets.pth")
    torch.save(best_model_wts, save_path)
    print(f"[+] Modele optimal enregistre : {save_path}")

    # 8. Courbes d'apprentissage
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(11, 4.5))

    # Perte
    plt.subplot(1, 2, 1)
    ep_range = range(1, num_epochs + 1)
    plt.plot(ep_range, history['train_loss'], 'b-', label='Train Loss', lw=2)
    plt.plot(ep_range, history['val_loss'], 'r--', label='Val Loss', lw=2)
    plt.axvline(x=best_epoch, color='green', linestyle=':', label=f'Meilleure epoque ({best_epoch})')
    plt.title(f"Courbe de Perte (Loss) - {architecture.upper()}", fontsize=11, fontweight='bold')
    plt.xlabel("Epoque")
    plt.ylabel("Perte (Cross-Entropy)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Exactitude
    plt.subplot(1, 2, 2)
    plt.plot(ep_range, [a * 100 for a in history['train_acc']], 'b-', label='Train Accuracy', lw=2)
    plt.plot(ep_range, [a * 100 for a in history['val_acc']], 'r--', label='Val Accuracy', lw=2)
    plt.axvline(x=best_epoch, color='green', linestyle=':', label=f'Meilleure epoque ({best_epoch})')
    plt.title(f"Courbe d'Exactitude (Accuracy) - {architecture.upper()}", fontsize=11, fontweight='bold')
    plt.xlabel("Epoque")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "courbes_apprentissage_cnn.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[+] Courbes sauvegardees : {plot_file}")

    # 9. JSON metrics
    metrics_file = os.path.join(output_dir, "metrics_cnn.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": architecture,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "total_time_seconds": total_time,
            "history": history
        }, f, indent=2)
    print(f"[+] Historique sauvegarde : {metrics_file}")

if __name__ == "__main__":
    train_cnn(architecture="resnet18", num_epochs=15)
