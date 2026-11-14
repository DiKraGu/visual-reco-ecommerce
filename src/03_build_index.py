import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import mobilenet_v2, resnet50

OUT_DIR = "data/processed"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

# Choisir "ResNet50" ou "MobileNetV2" selon les résultats de results/benchmark_metrics.json
MODEL_NAME = "ResNet50"

PREPROCESS_MAP = {
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "ResNet50": resnet50.preprocess_input,
}

if MODEL_NAME not in PREPROCESS_MAP:
    raise ValueError(f"Modèle non supporté : {MODEL_NAME}")

model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modèle introuvable : {model_path}")

print(f"Chargement du modèle {MODEL_NAME} et extraction des embeddings...")
catalog_df = pd.read_pickle(os.path.join(OUT_DIR, "full_catalog.pkl")).reset_index(drop=True)
full_model = models.load_model(model_path)

# Extraction de l'avant-dernière couche (vecteur dense avant Softmax)
feature_extractor = models.Model(
    inputs=full_model.input,
    outputs=full_model.get_layer("embedding_layer").output,
)

preprocess_fn = PREPROCESS_MAP[MODEL_NAME]


def build_inference_ds(paths: np.ndarray) -> tf.data.Dataset:
    def load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return preprocess_fn(img)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    return (
        ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


ds = build_inference_ds(catalog_df["image_path"].values)
embeddings = feature_extractor.predict(ds, verbose=1)

# Normalisation L2 : ||v||_2 = 1 => produit scalaire == similarité cosinus
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = (embeddings / np.maximum(norms, 1e-10)).astype(np.float32)

np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings_norm)
catalog_df.to_pickle(os.path.join(OUT_DIR, "indexed_catalog.pkl"))

with open(os.path.join(OUT_DIR, "index_config.json"), "w") as f:
    json.dump({"model_name": MODEL_NAME, "embedding_dim": int(embeddings_norm.shape[1])}, f, indent=4)

print(f"Indexation terminée : {embeddings_norm.shape[0]} produits, dimension {embeddings_norm.shape[1]}.")