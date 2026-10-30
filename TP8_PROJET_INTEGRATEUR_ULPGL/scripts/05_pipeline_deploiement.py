"""
05_pipeline_deploiement.py
GIN522 - Vision par Ordinateur & CNN (ULPGL 2025-2026)
Projet Integrateur - Pipeline de Deploiement & Demo Live
Auteur: Master 2 Genie Electrique et Informatique
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def build_model(num_classes=3):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

class BanknoteClassifier:
    def __init__(self, model_path=r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\models\best_resnet18_billets.pth"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['1000_fc', '500_fc', '50_fc']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = build_model(len(self.class_names))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[+] Modele charge avec succes ({self.device}) : {os.path.basename(model_path)}")
        else:
            raise FileNotFoundError(f"Fichier de poids introuvable : {model_path}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

    def predict(self, image_path, conf_threshold=0.60):
        t0 = time.time()
        orig_img = Image.open(image_path).convert("RGB")
        tensor_img = self.transform(orig_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = self.softmax(outputs).cpu().numpy()[0]

        latency_ms = (time.time() - t0) * 1000
        pred_idx = np.argmax(probs)
        conf = probs[pred_idx]
        pred_class = self.class_names[pred_idx]

        # Logique de securite metier (seuil d'incertitude)
        is_reliable = (conf >= conf_threshold)

        result = {
            "image": os.path.basename(image_path),
            "predicted_class": pred_class,
            "confidence": float(conf),
            "is_reliable": is_reliable,
            "latency_ms": latency_ms,
            "all_probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        }
        return result, orig_img

def run_demo(test_image=None):
    print("=" * 60)
    print("  ETAPE 7 : DEMONSTRATION LIVE (PIPELINE DE DEPLOIEMENT)")
    print("=" * 60)

    classifier = BanknoteClassifier()

    if test_image is None or not os.path.exists(test_image):
        test_dir = r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\dataset_split_224\test\1000_fc"
        candidates = os.listdir(test_dir)
        test_image = os.path.join(test_dir, candidates[0])

    print(f"\n[*] Analyse de l'image : {test_image}")
    res, img = classifier.predict(test_image)

    print("\n" + "-" * 45)
    print("RESULTAT DE L'INFERENCE EN TEMPS REEL")
    print("-" * 45)
    print(f"  Fichier analyse      : {res['image']}")
    print(f"  Classe Predite       : {res['predicted_class'].upper()}")
    print(f"  Score de Confiance   : {res['confidence']*100:.2f}%")
    print(f"  Temps de Calcul      : {res['latency_ms']:.1f} ms")
    print(f"  Statut Fiabilite     : {'CONFIRME (VALIDE)' if res['is_reliable'] else 'ALERTE : CONFIANCE FAIBLE !'}")
    print("\nDistribution complete des probabilites :")
    for k, v in res['all_probabilities'].items():
        print(f"  - {k.upper():<10} : {v*100:6.2f}%")

    # Annotation visuelle de la demo
    annotated = img.resize((640, 640)).copy()
    draw = ImageDraw.Draw(annotated)
    
    banner_color = (34, 139, 34) if res['is_reliable'] else (200, 30, 30)
    draw.rectangle([(0, 0), (640, 75)], fill=banner_color)
    
    text = f"PREDICTION : {res['predicted_class'].upper()} ({res['confidence']*100:.1f}%)"
    subtext = f"Temps : {res['latency_ms']:.1f} ms | {'VALIDE' if res['is_reliable'] else 'DOUTE / REVISION REQUISE'}"
    draw.text((20, 15), text, fill=(255, 255, 255))
    draw.text((20, 45), subtext, fill=(255, 255, 255))

    out_demo = r"E:\TP1_IMAGE_pROCESSING\TP8_PROJET_INTEGRATEUR_ULPGL\outputs\demo_prediction_live.png"
    annotated.save(out_demo)
    print(f"\n[+] Image annotee de la demo sauvegardee : {out_demo}")

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(img_arg)
