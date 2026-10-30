import os
import shutil
from pathlib import Path
from PIL import Image
import pillow_heif

# Register HEIF opener with PIL
pillow_heif.register_heif_opener()

SOURCE_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\dataset_billets")
TARGET_DIR = Path(r"E:\TP1_IMAGE_pROCESSING\dataset_billets_clean")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic'}
IGNORE_EXTENSIONS = {'.mov', '.mp4', '.aae'}

classes_map = {
    '50_fc': TARGET_DIR / '50_fc',
    '500_fc': TARGET_DIR / '500_fc',
    '1000_fc': TARGET_DIR / '1000_fc'
}

for c_target in classes_map.values():
    c_target.mkdir(parents=True, exist_ok=True)

stats = {}

for class_name, target_path in classes_map.items():
    src_class_dir = SOURCE_DIR / class_name
    count = 0
    errors = 0
    
    # Recursively find all files
    all_files = sorted(list(src_class_dir.rglob('*')))
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    ignored_files = [f for f in all_files if f.is_file() and f.suffix.lower() in IGNORE_EXTENSIONS]
    
    print(f"=== Processing Class: {class_name} ===")
    print(f"  Found {len(image_files)} candidate image files, {len(ignored_files)} ignored non-image files.")
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            with Image.open(img_path) as img:
                # Convert to RGB if needed (e.g. RGBA or CMYK)
                img_rgb = img.convert("RGB")
                target_file = target_path / f"{class_name}_{count+1:03d}.jpg"
                img_rgb.save(target_file, "JPEG", quality=95)
                count += 1
        except Exception as e:
            print(f"  [Error] Failed to convert {img_path.name}: {e}")
            errors += 1
            
    stats[class_name] = {'valid': count, 'errors': errors, 'ignored_videos_aae': len(ignored_files)}

print("\n=== DATASET SUMMARY ===")
for c, s in stats.items():
    print(f"Class: {c} -> {s['valid']} clean images saved, {s['errors']} errors, {s['ignored_videos_aae']} ignored videos/metadata")
