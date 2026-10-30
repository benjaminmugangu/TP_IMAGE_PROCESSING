"""
02_baseline_classique.py
GIN522 - Vision par Ordinateur & CNN (ULPGL 2025-2026)
Projet Integrateur - Etape Baseline Classique (Modules 1 a 4)
Auteur: Master 2 Genie Electrique et Informatique
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

def extract_handcrafted_features(image_path):
    """
    Extrait un vecteur de 30 caracteristiques classiques (Couleurs + Textures) :
    - Moyennes et ecarts-types RGB (6)
    - Moyennes et ecarts-types HSV (6)
    - Histogramme de Teinte H (16 bins)
    - Gradient spatial de Sobel (2 stats de texture)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Statistiques couleur RGB (6)
    mean_rgb = np.mean(img_rgb, axis=(0, 1))
    std_rgb = np.std(img_rgb, axis=(0, 1))

    # 2. Statistiques couleur HSV (6)
    mean_hsv = np.mean(img_hsv, axis=(0, 1))
    std_hsv = np.std(img_hsv, axis=(0, 1))

    # 3. Histogramme de Teinte (16 bins) normalise
    hist_h = cv2.calcHist([img_hsv], [0], None, [16], [0, 180])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()

    # 4. Textures / Contours (Sobel - Module 2-3)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    mean_grad = np.mean(magnitude)
    std_grad = np.std(magnitude)

    # Assemblage du vecteur de descripteurs (30 dimensions)
    features = np.hstack([
        mean_rgb, std_rgb,
        mean_hsv, std_hsv,
        hist_h,
        [mean_grad, std_grad]
    ])
    return features

def load_split_dataset(dataset_dir):
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    X, y, paths = [], [], []
    for cls in classes:
        cls_folder = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(cls_folder, fname)
                feats = extract_handcrafted_features(fpath)
                X.append(feats)
                y.append(class_to_idx[cls])
                paths.append(fpath)
                
    return np.array(X), np.array(y), classes, paths

def run_baseline_pipeline(
    data_root=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\dataset_split_224",
    output_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\outputs"
):
    print("=" * 60)
    print("  ETAPE 3 : BASELINE CLASSIQUE (Histogrammes HSV + Sobel + SVM/KNN)")
    print("=" * 60)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    print("[*] Extraction des descripteurs pour Train, Val et Test...")
    X_train, y_train, classes, _ = load_split_dataset(train_dir)
    X_val, y_val, _, _ = load_split_dataset(val_dir)
    X_test, y_test, _, test_paths = load_split_dataset(test_dir)

    print(f"[+] Dimensions des vecteurs extraits : {X_train.shape[1]} descripteurs par image")
    print(f"    - Train : {X_train.shape[0]} exemples")
    print(f"    - Val   : {X_val.shape[0]} exemples")
    print(f"    - Test  : {X_test.shape[0]} exemples")

    # Standardisation des descripteurs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. Modele A : KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_val_acc = accuracy_score(y_val, knn.predict(X_val_scaled))
    knn_test_pred = knn.predict(X_test_scaled)
    knn_test_acc = accuracy_score(y_test, knn_test_pred)
    knn_f1 = f1_score(y_test, knn_test_pred, average="macro")

    # 2. Modele B : SVM RBF (C=1.0)
    svm = SVC(kernel="rbf", C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_val_acc = accuracy_score(y_val, svm.predict(X_val_scaled))
    svm_test_pred = svm.predict(X_test_scaled)
    svm_test_acc = accuracy_score(y_test, svm_test_pred)
    svm_f1 = f1_score(y_test, svm_test_pred, average="macro")

    print("\n" + "-" * 50)
    print("RESULTATS DES BASELINES CLASSIQUES SUR LE TEST")
    print("-" * 50)
    print(f"1. KNN (k=5) : Accuracy = {knn_test_acc*100:.2f}% | Macro-F1 = {knn_f1*100:.2f}%")
    print(f"2. SVM (RBF) : Accuracy = {svm_test_acc*100:.2f}% | Macro-F1 = {svm_f1*100:.2f}%")

    # Selection de la meilleure baseline
    best_name = "SVM (RBF)" if svm_test_acc >= knn_test_acc else "KNN (k=5)"
    best_pred = svm_test_pred if best_name == "SVM (RBF)" else knn_test_pred
    best_acc = max(svm_test_acc, knn_test_acc)
    best_f1 = max(svm_f1, knn_f1)

    print(f"\n[+] Meilleure baseline retenue : {best_name} avec {best_acc*100:.2f}% de precision.")

    # Rapport detaille par classe
    report = classification_report(y_test, best_pred, target_names=classes, output_dict=True)
    print("\nDetail par classe (Meilleure baseline) :")
    for cls in classes:
        print(f"  Classe {cls:<10} : Precision={report[cls]['precision']*100:.1f}%, Rappel={report[cls]['recall']*100:.1f}%, F1={report[cls]['f1-score']*100:.1f}%")

    # Matrice de confusion
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.upper() for c in classes],
                yticklabels=[c.upper() for c in classes])
    plt.title(f"Matrice de Confusion Baseline ({best_name})\nAccuracy: {best_acc*100:.1f}%", fontsize=11, fontweight='bold')
    plt.xlabel("Prediction", fontweight='bold')
    plt.ylabel("Verite Terrain", fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "matrice_confusion_baseline.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[+] Matrice de confusion enregistree : {cm_path}")

    # Sauvegarde JSON
    metrics_path = os.path.join(output_dir, "metrics_baseline.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": best_name,
            "test_accuracy": best_acc,
            "test_macro_f1": best_f1,
            "knn_accuracy": knn_test_acc,
            "svm_accuracy": svm_test_acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }, f, indent=2)
    print(f"[+] Metriques baseline sauvegardees : {metrics_path}")

if __name__ == "__main__":
    run_baseline_pipeline()
