# PROJET INTÉGRATEUR : SYSTÈME DE VISION INDUSTRIEL
## Reconnaissance et Authentification de Billets en Francs Congolais (CDF)
### Master 2 Génie Électrique & Informatique — ULPGL (2025-2026)
**Auteur :** MUGANGU MUGISHO Benjamin  
**Enseignant :** Dr. KAMBALE WAMUHINDO Abednego  


---

## 📌 Présentation du Projet
Ce projet répond aux exigences du **Module 8 (Traitement d'Images & Vision par Ordinateur)**. Il développe un système complet et embarquable de reconnaissance automatique des billets de banque circulant en République Démocratique du Congo :
* **50 FC** (100 images)
* **500 FC** (106 images)
* **1000 FC** (100 images)
Total : **306 images réelles**.

---

## 🚀 Structure du Répertoire
```text
TP8_PROJET_INTEGRATEUR_ULPGL/
│
├── TP8_Projet_Integrateur_Vision_ULPGL.ipynb  <-- Notebook complet documenté et exécuté
├── rapport_tp8.tex                           <-- Rapport de synthèse en LaTeX (10 p. max)
├── GUIDE_ANNOTATION.md                       <-- Guide d'annotation formel et contrôle qualité
├── README.md                                 <-- Guide d'exécution et de reproduction
│
├── scripts/                                  <-- Scripts modulaires Python
│   ├── 01_split_et_hygiene.py                <-- Découpage 70/15/15 sans fuite (seed=42)
│   ├── 02_baseline_classique.py              <-- Baseline couleur HSV + Sobel + SVM/KNN
│   ├── 03_train_cnn_transfer.py              <-- Transfer Learning ResNet-18 + Data Augmentation
│   ├── 04_evaluation_autopsie.py             <-- Évaluation honnête test + autopsie d'erreurs
│   ├── 05_pipeline_deploiement.py            <-- Inférence en temps réel (démo live)
│   └── build_tp8_notebook.py                 <-- Assembleur automatique du notebook
│
├── models/
│   └── best_resnet18_billets.pth             <-- Poids du meilleur modèle sauvegardé
│
├── outputs/                                  <-- Tous les graphiques et métriques
│   ├── repartition_donnees.png
│   ├── matrice_confusion_baseline.png
│   ├── courbes_apprentissage_cnn.png
│   ├── comparaison_baseline_vs_cnn.png
│   ├── autopsie_borderline_ou_erreurs.png
│   ├── demo_prediction_live.png
│   └── metrics_comparatif_final.json
│
└── dataset_split_224/                        <-- Jeux de données redimensionnés étanches
    ├── train/ (214 images)
    ├── val/   (45 images)
    └── test/  (47 images)
```

---

## ⚙️ Reproduction Pas-à-Pas (En Ligne de Commande)

Pour reproduire l'intégralité du pipeline sur votre machine :

```bash
# 1. Découpage stratifié et préparation sans fuite
python scripts/01_split_et_hygiene.py

# 2. Entraînement et évaluation de la baseline classique
python scripts/02_baseline_classique.py

# 3. Entraînement du réseau de neurones par Transfer Learning
python scripts/03_train_cnn_transfer.py

# 4. Évaluation sur les 47 images de test inconnues & autopsie critique
python scripts/04_evaluation_autopsie.py

# 5. Démonstration en direct (inférence live sur une photo)
python scripts/05_pipeline_deploiement.py "dataset_split_224/test/1000_fc/1000_fc_004.jpg"
```

---

## 📊 Résultats Clés de l'Évaluation Scientifique

| Métrique | Baseline Classique (SVM) | CNN ResNet-18 (Transfer Learning) | Gain Net |
| :--- | :---: | :---: | :---: |
| **Exactitude (*Accuracy*)** | **89.36 %** | **100.00 %** | **+10.64 %** |
| **Macro F1-Score** | **89.16 %** | **100.00 %** | **+10.84 %** |
| **Latence CPU** | ~2 ms | **122.2 ms** | Temps réel garanti ($<200$ ms) |
| **Taille Modèle** | ~5 Ko | **46.8 Mo** | 100 % hors-ligne (sans Internet) |
