# Système de Recommandation Visuelle E-Commerce (CNN & Transfer Learning)

Moteur de recherche par similarité visuelle (*Content-Based Image Retrieval*) appliqué au commerce électronique. À partir d'une image produit (issue du catalogue ou importée par l'utilisateur), le système extrait une représentation vectorielle profonde (*embedding* de dimension 2048 via **ResNet50**) et retourne les articles visuellement les plus proches par similarité cosinus, combinés à un filtrage hybride sur les métadonnées métier (genre, catégorie, sous-catégorie, couleur).

---

## Architecture du projet

```text
visual-reco-ecommerce/
├── data/
│   ├── images/                  # 8 287 images produits extraites (ignoré par Git)
│   └── processed/               # Splits stratifiés (.pkl), embeddings L2 (.npy) et config JSON (ignoré par Git)
├── models/                      # Modèles entraînés MobileNetV2.keras et ResNet50.keras (ignoré par Git)
├── results/                     # Courbes d'apprentissage, matrices de confusion et benchmark_metrics.json
├── src/
│   ├── 01_prepare_data.py       # Extraction depuis archive.zip, équilibrage (30 classes) et split 70/15/15
│   ├── 02_train_models.py       # Entraînement, fine-tuning et évaluation (MobileNetV2, ResNet50, VGG19)
│   ├── 03_build_index.py        # Extraction des embeddings ResNet50 (2048-D) et normalisation L2
│   └── recommender.py           # Moteur d'inférence, filtrage hybride et calcul de similarité cosinus
├── app.py                       # Interface web interactive Streamlit
├── requirements.txt             # Dépendances Python exactes
└── README.md
```

---

## Installation et configuration

### 1. Cloner le dépôt et créer l'environnement virtuel

```bash
git clone <URL_DU_DEPOT_GITHUB>
cd visual-reco-ecommerce

python -m venv venv
```

**Activation de l'environnement virtuel :**
* **Windows (PowerShell) :**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (CMD) :**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
* **Linux / macOS :**
  ```bash
  source venv/bin/activate
  ```

### 2. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Pipeline d'exécution complet

### Prérequis : Jeu de données
Télécharger l'archive Kaggle **Fashion Product Images (Small)** et placer le fichier `archive.zip` dans votre dossier de téléchargements (`~/Downloads/archive.zip` ou `C:\Users\<Utilisateur>\Downloads\archive.zip`).

### Étape 1 : Préparation des données et split stratifié
Lit directement `archive.zip`, filtre les sous-catégories contenant au moins 50 produits, plafonne chaque classe à 350 images (8 287 produits sur 30 catégories) et génère les partitions stratifiées Train (70 %), Validation (15 %) et Test (15 %).

```bash
python src/01_prepare_data.py
```
*(Optionnel : une fois cette étape terminée, `archive.zip` peut être supprimé pour libérer de l'espace disque).*

### Étape 2 : Entraînement et benchmark des architectures CNN
Entraîne et évalue séquentiellement **MobileNetV2**, **ResNet50** (Feature Extraction + Fine-Tuning des 20 dernières couches) et **VGG19** (Feature Extraction pur), puis sauvegarde les courbes d'apprentissage, les matrices de confusion $30 \times 30$ et `benchmark_metrics.json` dans `results/`.

```bash
python src/02_train_models.py
```

### Étape 3 : Indexation vectorielle L2
Extrait les vecteurs caractéristiques (couche `GlobalAveragePooling2D`, dimension 2048) de l'ensemble du catalogue avec **ResNet50** et applique une normalisation $L_2$ ($\Vert{}\hat{v}\Vert{}_2 = 1$) afin d'assimiler le produit scalaire à la similarité cosinus.

```bash
python src/03_build_index.py
```

### Étape 4 : Lancement de l'application web temps réel

```bash
streamlit run app.py
```
L'interface est accessible localement à l'adresse : `http://localhost:8501`.

---

## Résultats du benchmark (Jeu de Test — 1 244 images, 30 catégories)

| Architecture | Protocole d'entraînement | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Temps CPU | Taille disque |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** *(retenu)* | Feature Extraction + Fine-Tuning (20 couches) | **90,84 %** | **86,90 %** | **85,89 %** | **86,13 %** | 3370,3 s (56 min) | 159,54 Mo |
| **MobileNetV2** | Feature Extraction + Fine-Tuning (20 couches) | 89,23 % | 86,17 % | 85,79 % | 85,77 % | **1512,5 s (25 min)** | **18,88 Mo** |
| **VGG19** | Feature Extraction pur (base gelée) | 86,01 % | 83,73 % | 81,45 % | 81,08 % | 3039,3 s (50 min) | 76,68 Mo |

---

## Fonctionnalités de l'interface (`app.py`)
* **Double mode de requête :** sélection aléatoire d'un article du catalogue (vérification de l'identité à 100 %) ou importation d'une image externe (`JPG`, `JPEG`, `PNG`, `WEBP`).
* **Filtrage hybride dynamique :** restriction des résultats par `Genre`, `Catégorie principale`, `Sous-catégorie` et `Couleur dominante`.
* **Inférence temps réel :** mesure de la latence globale (extraction CNN + produit matriciel) affichée en millisecondes (~60 à 180 ms sur CPU).

## Stack technique
* **Deep Learning :** TensorFlow / Keras (MobileNetV2, ResNet50, VGG19 pré-entraînés sur ImageNet)
* **Traitement de données & Calcul matriciel :** NumPy, Pandas, Scikit-learn, Pillow
* **Interface & Visualisation :** Streamlit, Matplotlib, Seaborn