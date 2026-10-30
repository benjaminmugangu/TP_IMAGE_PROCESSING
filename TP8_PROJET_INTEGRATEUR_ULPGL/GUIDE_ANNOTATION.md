# GUIDE MÉTHODOLOGIQUE D'ANNOTATION ET CONTRÔLE QUALITÉ
## Projet Intégrateur — Système de Vision pour Billets de Banque Congolais (CDF)
### ULPGL — Master 2 Génie Électrique & Informatique (GIN522)
**Auteur :** MUGANGU MUGISHO Benjamin  
**Enseignant :** Dr. KAMBALE WAMUHINDO Abednego  


---

## 1. Objectif du Guide
Ce document formalise les règles strictes d'inclusion, d'exclusion et de labellisation des photographies de billets de Francs Congolais (50 FC, 500 FC, 1000 FC). Son application garantit la cohérence inter-annotateurs et prévient l'apprentissage de biais parasites (*shortcut learning*).

---

## 2. Règles d'Inclusion et d'Étiquetage des Classes

| Classe | Coupure | Couleurs Clés | Motifs Distinctifs Principaux |
| :--- | :--- | :--- | :--- |
| `50_fc` | 50 Francs Congolais | Ocre, brun, orangé | Masque tshokwe au recto, pêcheurs au verso, chiffre « 50 » |
| `500_fc` | 500 Francs Congolais | Bleu ciel, cyan, gris-bleu | Pont Matadi au recto, faune au verso, chiffre « 500 » |
| `1000_fc` | 1000 Francs Congolais | Vert foncé, bleu sombre | Perroquet gris, armoiries, maïs au verso, chiffre « 1000 » |

---

## 3. Gestion des Cas Particuliers de Terrain (Protocole de Décision)

### A. Billets Coupés ou Partiellement Visibles
* **Règle :** Le billet doit présenter au minimum **70 % de sa surface** dans le champ de la caméra.
* **Critère discriminant :** Au moins un des indicateurs de valeur (« 50 », « 500 », « 1000 ») ou le portrait/motif principal doit être entièrement visible.
* **Rejet :** Si moins de 70 % est visible, l'image est rejetée pour la classification globale.

### B. Billets Flous et Problèmes de Mise au Point
* **Règle :** Un léger flou de bougé est toléré pour simuler la prise de vue smartphone au marché.
* **Rejet :** Si les micro-gravures et les chiffres sont tellement flous que même un être humain hésite entre 500 et 1000 FC, l'image est écartée.

### C. Billets Très Usés, Pliés, Déchirés ou Scotchés
* **Règle :** Les billets scotchés avec ruban adhésif transparent sont **acceptés** (ils reflètent la réalité économique locale de Goma).
* **Ruban opaque :** Si le ruban est opaque et masque le chiffre de la coupure, classer d'après le motif résiduel visible ; si le doute subsiste, rejeter.

### D. Variabilité des Arrière-Plans et Éclairages
* Pour éviter le biais d'arrière-plan (*background bias*), chaque classe de billet doit comporter :
  - Au moins 3 types de supports différents (table en bois, comptoir métallique/carrelé, tissu/nappe).
  - Au moins 2 conditions d'éclairage (lumière naturelle du jour, lumière artificielle intérieure/tamisée).

---

## 4. Protocole de Contrôle Qualité et Accord Inter-Annotateurs
Conformément aux directives du cours (diapositive 12) :
1. **Double annotation :** 10 % des images (soit 31 images) sont annotées de manière indépendante par deux personnes distinctes.
2. **Mesure de concordance :** Taux d'accord = 100 % sur les labels de coupure.
3. **Plafond humain :** Toute divergence humaine sur une coupure trop altérée entraîne le reclassement de l'image en statut « cas limite / rejet de confiance ».
