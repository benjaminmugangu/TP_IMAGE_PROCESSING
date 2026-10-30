"""
01_split_et_hygiene.py
GIN522 - Vision par Ordinateur & CNN (ULPGL 2025-2026)
Projet Integrateur - Etape Donnees et Hygiene Experimentale
Auteur: Master 2 Genie Electrique et Informatique
"""

import os
import shutil
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

def split_and_prepare_dataset(
    source_dir=r"E:\TP1_IMAGE_pROCESSING\TP7_CNN_CLASSIFICATION_ULPGL\dataset_billets_clean",
    dest_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\dataset_split",
    dest_dir_224=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\dataset_split_224",
    output_dir=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\outputs",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    print("=" * 60)
    print("  ETAPE 2 : DECOUPAGE STRATIFIE ET HYGIENE EXPERIMENTALE")
    print("=" * 60)

    # 1. Verification de la graine aleatoire
    random.seed(seed)
    print(f"[+] Graine aleatoire fixee : seed={seed} (reproductibilite garantie)")

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    classes.sort()
    print(f"[+] Classes detectees ({len(classes)}) : {classes}")

    manifest = {"train": {}, "val": {}, "test": {}, "total_per_class": {}}
    stats = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
            os.makedirs(os.path.join(dest_dir_224, split, cls), exist_ok=True)
            stats[split][cls] = 0

    total_images_processed = 0

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()  # ordre deterministe avant melange
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Le reste va dans le test pour garantir sum == n_total
        n_test = n_total - n_train - n_val

        splits_files = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        manifest["total_per_class"][cls] = n_total

        print(f"\n[*] Classe [{cls}] : {n_total} images")
        print(f"    - Train : {len(splits_files['train'])} ({len(splits_files['train'])/n_total*100:.1f}%)")
        print(f"    - Val   : {len(splits_files['val'])} ({len(splits_files['val'])/n_total*100:.1f}%)")
        print(f"    - Test  : {len(splits_files['test'])} ({len(splits_files['test'])/n_total*100:.1f}%)")

        for split_name, flist in splits_files.items():
            manifest[split_name][cls] = flist
            stats[split_name][cls] = len(flist)

            for fname in flist:
                src_f = os.path.join(cls_path, fname)
                dst_f = os.path.join(dest_dir, split_name, cls, fname)
                dst_f_224 = os.path.join(dest_dir_224, split_name, cls, fname)

                # Copie originale
                shutil.copy2(src_f, dst_f)

                # Pre-redimensionnement standardise 224x224 RGB (anti-saturation RAM/Colab)
                with Image.open(src_f) as img:
                    img_rgb = img.convert("RGB")
                    img_resized = img_rgb.resize((224, 224), Image.Resampling.LANCZOS)
                    img_resized.save(dst_f_224, "JPEG", quality=95)

                total_images_processed += 1

    # 2. Sauvegarde du Manifest JSON
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "classes": classes,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "stats": stats,
            "manifest": manifest
        }, f, indent=2)
    print(f"\n[+] Manifeste de tracabilite sauvegarde : {manifest_path}")

    # 3. Graphique professionnel de la repartition
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(classes))
    width = 0.25

    train_vals = [stats["train"][c] for c in classes]
    val_vals = [stats["val"][c] for c in classes]
    test_vals = [stats["test"][c] for c in classes]

    bars1 = ax.bar([i - width for i in x], train_vals, width, label='Train (70%)', color='#2b5c8f')
    bars2 = ax.bar(x, val_vals, width, label='Validation (15%)', color='#e08214')
    bars3 = ax.bar([i + width for i in x], test_vals, width, label='Test (15%)', color='#4daf4a')

    # Ajout des valeurs au dessus des barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel("Nombre d'images", fontsize=11, fontweight='bold')
    ax.set_title("Projet Integrateur ULPGL - Repartition Stratifiee du Dataset (CDF)", fontsize=12, fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Billet {c.replace('_', ' ').upper()}" for c in classes], fontsize=10, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plot_path = os.path.join(output_dir, "repartition_donnees.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Graphique de repartition sauvegarde : {plot_path}")
    print("\n[SUCCESS] Etape de decoupage terminee sans aucune fuite d'information !")

if __name__ == "__main__":
    split_and_prepare_dataset()
