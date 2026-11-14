import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

ZIP_PATH = os.path.expanduser(r"~\Downloads\archive.zip")
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
OUT_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"Archive introuvable : {ZIP_PATH}")

print(f"Lecture directe de l'archive : {ZIP_PATH}")

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    all_files = set(zf.namelist())
    csv_member = "styles.csv" if "styles.csv" in all_files else "myntradataset/styles.csv"
    img_prefix = "images/" if "styles.csv" in all_files else "myntradataset/images/"

    # 1. Lecture du CSV
    with zf.open(csv_member) as f:
        df = pd.read_csv(f, on_bad_lines="skip")

    cols = ["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour"]
    df = df.dropna(subset=cols).copy()

    # 2. Vérifier la présence de l'image dans le ZIP
    df["zip_member"] = df["id"].apply(lambda x: f"{img_prefix}{x}.jpg")
    df = df[df["zip_member"].isin(all_files)].copy()

    # 3. Filtrer les catégories >= 50 produits et plafonner à 350 par catégorie (~7 500 images)
    counts = df["subCategory"].value_counts()
    valid_classes = counts[counts >= 50].index
    df = df[df["subCategory"].isin(valid_classes)].copy()

    df = (
        df.sample(frac=1.0, random_state=42)
        .groupby("subCategory", as_index=False)
        .head(350)
        .reset_index(drop=True)
    )

    # 4. Extraction uniquement des images manquantes parmi les ~7 500 retenues
    print(f"Vérification/extraction de {len(df)} images sélectionnées...")
    local_paths = []
    for _, row in df.iterrows():
        dest_path = os.path.join(IMG_DIR, f"{row['id']}.jpg")
        if not os.path.exists(dest_path):
            with open(dest_path, "wb") as out_img:
                out_img.write(zf.read(row["zip_member"]))
        local_paths.append(dest_path)

    df["image_path"] = local_paths
    df = df.drop(columns=["zip_member"])

# 5. Nettoyage des images excédentaires éventuellement extraites lors du test précédent
kept_files = set(os.path.basename(p) for p in df["image_path"])
removed = 0
for fname in os.listdir(IMG_DIR):
    if fname not in kept_files:
        os.remove(os.path.join(IMG_DIR, fname))
        removed += 1

# 6. Split stratifié : 70% Train, 15% Validation, 15% Test
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["subCategory"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["subCategory"])

df.to_pickle(os.path.join(OUT_DIR, "full_catalog.pkl"))
train_df.to_pickle(os.path.join(OUT_DIR, "train.pkl"))
val_df.to_pickle(os.path.join(OUT_DIR, "val.pkl"))
test_df.to_pickle(os.path.join(OUT_DIR, "test.pkl"))

print(f"Terminé : {len(df)} produits ({len(valid_classes)} catégories) prêts dans {OUT_DIR}. ({removed} images excédentaires nettoyées).")