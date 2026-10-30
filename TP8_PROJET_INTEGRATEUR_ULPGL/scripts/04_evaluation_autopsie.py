"""
04_evaluation_autopsie.py
GIN522 - Vision par Ordinateur & CNN (ULPGL 2025-2026)
Projet Integrateur - Etape Evaluation Finale & Autopsie des Erreurs
Auteur: Master 2 Genie Electrique et Informatique
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from PIL import Image

def get_model(architecture="resnet18", num_classes=3):
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    return model

def run_evaluation_and_autopsy(
    data_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\dataset_split_224",
    model_path=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\models\best_resnet18_billets.pth",
    baseline_metrics_path=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\outputs\metrics_baseline.json",
    output_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\outputs"
):
    print("=" * 60)
    print("  ETAPE 6 : EVALUATION FINALE SUR LE TEST ET AUTOPSIE CRITIQUE")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Materiel de calcul : {device}")

    # 1. Pipeline de Test
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"[+] Classes ({num_classes}) : {class_names}")
    print(f"[+] Nombre d'images de test : {len(test_dataset)}")

    # 2. Chargement du Modele
    print(f"\n[*] Chargement des poids optimaux depuis : {model_path}")
    model = get_model(architecture="resnet18", num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Inferences sur le jeu de test
    y_true = []
    y_pred = []
    confidences = []
    probabilities = []
    image_paths = [s[0] for s in test_dataset.samples]

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            conf = probs[pred_idx]

            y_true.append(labels.item())
            y_pred.append(pred_idx)
            confidences.append(conf)
            probabilities.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    # 4. Calcul des Metriques Globales
    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print("\n" + "=" * 50)
    print(f"  PERFORMANCES FINALES SUR LE TEST (CNN RESNET18)")
    print("=" * 50)
    print(f"  Exactitude Globale (Accuracy) : {test_acc*100:.2f}%")
    print(f"  Macro-F1 Score                : {macro_f1*100:.2f}%")
    print(f"  Weighted-F1 Score             : {weighted_f1*100:.2f}%")
    print("-" * 50)
    for cls in class_names:
        print(f"  Classe {cls:<10} : Precision={report[cls]['precision']*100:.1f}%, Rappel={report[cls]['recall']*100:.1f}%, F1={report[cls]['f1-score']*100:.1f}%")

    # 5. Comparaison avec la Baseline Classique
    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    base_acc = baseline_metrics["test_accuracy"]
    base_f1 = baseline_metrics["test_macro_f1"]
    base_name = baseline_metrics["model"]

    print("\n" + "=" * 50)
    print("  COMPARATIF SCIENTIFIQUE : BASELINE vs CNN")
    print("=" * 50)
    print(f"  Baseline ({base_name}) : Accuracy = {base_acc*100:.2f}% | F1 = {base_f1*100:.2f}%")
    print(f"  CNN (ResNet18)        : Accuracy = {test_acc*100:.2f}% | F1 = {macro_f1*100:.2f}%")
    gain_acc = (test_acc - base_acc) * 100
    print(f"  --> Gain apporte par le Deep Learning : +{gain_acc:.2f} points de pourcentage !")

    # 6. Graphique Comparatif & Matrice de Confusion
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Matrice de confusion CNN
    cm_cnn = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=[c.upper() for c in class_names],
                yticklabels=[c.upper() for c in class_names])
    axes[0].set_title(f"Matrice de Confusion CNN ResNet18\nAccuracy Test : {test_acc*100:.2f}%", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Prediction du CNN", fontweight='bold')
    axes[0].set_ylabel("Verite Terrain", fontweight='bold')

    # Barplot Comparatif Baseline vs CNN
    models_labels = [f"Baseline\n({base_name})", "CNN\n(ResNet18)"]
    acc_scores = [base_acc * 100, test_acc * 100]
    f1_scores = [base_f1 * 100, macro_f1 * 100]

    x = np.arange(len(models_labels))
    w = 0.35
    b1 = axes[1].bar(x - w/2, acc_scores, w, label='Accuracy (%)', color='#2b5c8f')
    b2 = axes[1].bar(x + w/2, f1_scores, w, label='Macro-F1 (%)', color='#4daf4a')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            axes[1].annotate(f'{h:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1].set_title("Comparatif des Performances (Sur 47 images de Test)", fontsize=11, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_labels, fontweight='bold')
    axes[1].set_ylim(0, 115)
    axes[1].legend(frameon=True)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    comp_path = os.path.join(output_dir, "comparaison_baseline_vs_cnn.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"\n[+] Graphique comparatif sauvegarde : {comp_path}")

    # 7. Autopsie : Recherche des erreurs ou des cas a plus faible confiance
    errors = np.where(y_true != y_pred)[0]
    print(f"\n[*] Nombre d'erreurs pures sur le test : {len(errors)} / {len(y_true)}")

    # Selection des 6 cas les plus critiques (erreurs ou confiances les plus basses)
    if len(errors) > 0:
        critical_indices = errors[:6]
    else:
        # Si 100% de reussite, on analyse les 6 images ou le reseau a ete le moins sur de lui (confiance minimale)
        critical_indices = np.argsort(confidences)[:6]
        print("[+] Aucune erreur pure ! Analyse des 6 predictions a confiance la plus faible (cas limites).")

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(critical_indices):
        img_path = image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        true_cls = class_names[y_true[idx]]
        pred_cls = class_names[y_pred[idx]]
        conf = confidences[idx]

        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        color = 'green' if true_cls == pred_cls else 'red'
        plt.title(f"Reel: {true_cls.upper()}\nPred: {pred_cls.upper()} ({conf*100:.1f}%)",
                  fontsize=9, fontweight='bold', color=color)

    plt.suptitle("Autopsie Critique : Cas Limites et Analyse Qualite (Jeu de Test)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    autopsy_path = os.path.join(output_dir, "autopsie_borderline_ou_erreurs.png")
    plt.savefig(autopsy_path, dpi=300)
    plt.close()
    print(f"[+] Planche d'autopsie critique sauvegardee : {autopsy_path}")

    # 8. Sauvegarde du bilan final
    final_json_path = os.path.join(output_dir, "metrics_comparatif_final.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_samples": len(test_dataset),
            "baseline": {
                "name": base_name,
                "accuracy": base_acc,
                "macro_f1": base_f1
            },
            "cnn_resnet18": {
                "accuracy": test_acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "classification_report": report,
                "confusion_matrix": cm_cnn.tolist()
            },
            "gain_accuracy_points": gain_acc
        }, f, indent=2)
    print(f"[+] Bilan comparatif JSON sauvegarde : {final_json_path}")
    print("\n[SUCCESS] Etape 6 terminee avec succes !")

if __name__ == "__main__":
    run_evaluation_and_autopsy()
