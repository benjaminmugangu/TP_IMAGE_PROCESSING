import os
import cv2
from pathlib import Path

BASE_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\TP7_CNN_CLASSIFICATION_ULPGL")
SRC_DIR = BASE_DIR / "dataset_split"
DST_DIR = BASE_DIR / "dataset_split_fast"

TARGET_SIZE = (256, 256)

print(f"=== PRÉ-TRAITEMENT ET OPTIMISATION DES IMAGES ({TARGET_SIZE[0]}x{TARGET_SIZE[1]}) ===")

for split in ['train', 'val', 'test']:
    for cls in ['50_fc', '500_fc', '1000_fc']:
        src_cls = SRC_DIR / split / cls
        dst_cls = DST_DIR / split / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        
        images = list(src_cls.glob("*.jpg"))
        for img_p in images:
            img = cv2.imread(str(img_p))
            if img is not None:
                img_res = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_cls / img_p.name), img_res, [cv2.IMWRITE_JPEG_QUALITY, 92])

print("[OK] Pré-traitement terminé avec succès dans dataset_split_fast !")
