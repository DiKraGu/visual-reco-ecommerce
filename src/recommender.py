import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import models
from tensorflow.keras.applications import mobilenet_v2, resnet50

PREPROCESS_MAP = {
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "ResNet50": resnet50.preprocess_input,
}


class VisualRecommender:
    def __init__(self, processed_dir: str = "data/processed", model_dir: str = "models"):
        config_path = os.path.join(processed_dir, "index_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        self.model_name = config["model_name"]
        self.preprocess_fn = PREPROCESS_MAP[self.model_name]

        model_path = os.path.join(model_dir, f"{self.model_name}.keras")
        full_model = models.load_model(model_path)

        self.extractor = models.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer("embedding_layer").output,
        )
        self.embeddings = np.load(os.path.join(processed_dir, "embeddings.npy"))
        self.catalog = pd.read_pickle(os.path.join(processed_dir, "indexed_catalog.pkl"))

    def extract_query_embedding(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        arr = self.preprocess_fn(arr)

        vec = self.extractor.predict(arr, verbose=0)[0]
        norm = np.linalg.norm(vec)
        return (vec / max(norm, 1e-10)).astype(np.float32)

    def search(self, pil_image: Image.Image, top_k: int = 8, filters: dict | None = None):
        t0 = time.perf_counter()
        query_vec = self.extract_query_embedding(pil_image)

        # Filtrage hybride sur les métadonnées métier
        mask = np.ones(len(self.catalog), dtype=bool)
        if filters:
            for col, val in filters.items():
                if val and val != "Tous":
                    mask &= (self.catalog[col] == val).values

        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            return pd.DataFrame(), 0.0

        # Calcul vectorisé de la similarité cosinus
        sub_embeddings = self.embeddings[filtered_indices]
        similarities = np.dot(sub_embeddings, query_vec)

        k = min(top_k, len(similarities))
        top_local_idx = np.argsort(similarities)[::-1][:k]
        top_global_idx = filtered_indices[top_local_idx]

        results = self.catalog.iloc[top_global_idx].copy()
        results["similarity_score"] = similarities[top_local_idx]
        latency_ms = (time.perf_counter() - t0) * 1000

        return results, latency_ms