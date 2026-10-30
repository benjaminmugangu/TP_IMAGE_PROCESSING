import os
import shutil
import random
from pathlib import Path

# Fixer la graine aléatoire pour une reproductibilité scientifique absolue
random.seed(42)

CLEAN_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\dataset_billets_clean")
SPLIT_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\dataset_split")

# Création des répertoires cibles
for split in ['train', 'val', 'test']:
    for cls in ['50_fc', '500_fc', '1000_fc']:
        (SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

classes = ['50_fc', '500_fc', '1000_fc']

print("=== DÉCOUPAGE REPRODUCTIBLE 70% TRAIN / 15% VAL / 15% TEST ===")

summary = {}

for cls in classes:
    cls_dir = CLEAN_DIR / cls
    all_images = sorted(list(cls_dir.glob("*.jpg")))
    
    # Mélange déterministe avec seed 42
    random.shuffle(all_images)
    n = len(all_images)
    
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    # Le reste va dans test pour garantir que train + val + test == n
    n_test = n - n_train - n_val
    
    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train + n_val]
    test_imgs = all_images[n_train + n_val:]
    
    for img in train_imgs:
        shutil.copy2(img, SPLIT_DIR / 'train' / cls / img.name)
    for img in val_imgs:
        shutil.copy2(img, SPLIT_DIR / 'val' / cls / img.name)
    for img in test_imgs:
        shutil.copy2(img, SPLIT_DIR / 'test' / cls / img.name)
        
    summary[cls] = {
        'total': n,
        'train': len(train_imgs),
        'val': len(val_imgs),
        'test': len(test_imgs)
    }
    print(f"Classe {cls:>7} : {n} images -> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

print("-" * 55)
total_train = sum(s['train'] for s in summary.values())
total_val = sum(s['val'] for s in summary.values())
total_test = sum(s['test'] for s in summary.values())
total_all = sum(s['total'] for s in summary.values())
print(f"TOTAL GLOBAL   : {total_all} images -> Train: {total_train} | Val: {total_val} | Test: {total_test}")
