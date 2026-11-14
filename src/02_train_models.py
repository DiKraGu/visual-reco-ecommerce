import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG19
from tensorflow.keras.applications import mobilenet_v2, resnet50, vgg19
from sklearn.metrics import classification_report, confusion_matrix

OUT_DIR = "data/processed"
MODEL_DIR = "models"
RES_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_df = pd.read_pickle(os.path.join(OUT_DIR, "train.pkl"))
val_df = pd.read_pickle(os.path.join(OUT_DIR, "val.pkl"))
test_df = pd.read_pickle(os.path.join(OUT_DIR, "test.pkl"))

classes = sorted(train_df["subCategory"].unique())
num_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}

# Tuple : (Classe du modèle, fonction de prétraitement ImageNet, activer le fine-tuning)
# VGG19 reste gelé sur CPU pour éviter plusieurs heures de rétropropagation sur ses 143M de paramètres
BACKBONES = {
    "MobileNetV2": (MobileNetV2, mobilenet_v2.preprocess_input, True),
    "ResNet50": (ResNet50, resnet50.preprocess_input, True),
    "VGG19": (VGG19, vgg19.preprocess_input, False),
}

def make_dataset(df, preprocess_fn, is_train=False):
    paths = df["image_path"].values
    labels = df["subCategory"].map(class_to_idx).values

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)  # Valeurs float32 dans [0, 255]
        if is_train:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=20.0)
            img = tf.clip_by_value(img, 0.0, 255.0)
        img = preprocess_fn(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_train:
        ds = ds.shuffle(buffer_size=1024)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_model(backbone_cls):
    base = backbone_cls(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="embedding_layer")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base

comparison_metrics = {}

for name, (backbone_cls, preprocess_fn, do_finetune) in BACKBONES.items():
    print(f"\n=== Entraînement : {name} ===")
    t_start = time.time()

    train_ds = make_dataset(train_df, preprocess_fn, is_train=True)
    val_ds = make_dataset(val_df, preprocess_fn, is_train=False)
    test_ds = make_dataset(test_df, preprocess_fn, is_train=False)

    model, base = build_model(backbone_cls)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]

    # Phase 1 : Entraînement de la tête de classification
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=cb)

    acc = list(h1.history["accuracy"])
    val_acc = list(h1.history["val_accuracy"])
    loss = list(h1.history["loss"])
    val_loss = list(h1.history["val_loss"])

    # Phase 2 : Fine-Tuning partiel (dégel des 20 dernières couches)
    if do_finetune:
        print(f"--- Fine-Tuning ({name}) ---")
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=cb)
        acc += h2.history["accuracy"]
        val_acc += h2.history["val_accuracy"]
        loss += h2.history["loss"]
        val_loss += h2.history["val_loss"]

    train_duration = round(time.time() - t_start, 1)
    model_path = os.path.join(MODEL_DIR, f"{name}.keras")
    model.save(model_path)
    model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)

    # Évaluation sur le jeu de Test
    y_true = test_df["subCategory"].map(class_to_idx).values
    y_probs = model.predict(test_ds)
    y_pred = np.argmax(y_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    comparison_metrics[name] = {
        "accuracy": round(report["accuracy"], 4),
        "precision_macro": round(report["macro avg"]["precision"], 4),
        "recall_macro": round(report["macro avg"]["recall"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "train_time_sec": train_duration,
        "model_size_mb": model_size_mb,
    }

    # Sauvegarde des courbes Loss & Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="Train Acc")
    axes[0].plot(val_acc, label="Val Acc")
    axes[0].set_title(f"{name} - Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()

    axes[1].plot(loss, label="Train Loss")
    axes[1].plot(val_loss, label="Val Loss")
    axes[1].set_title(f"{name} - Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f"{name}_curves.png"), dpi=150)
    plt.close()

    # Sauvegarde de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", annot=True, fmt="d", annot_kws={"size": 7})
    plt.title(f"Matrice de Confusion - {name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f"{name}_cm.png"), dpi=150)
    plt.close()

with open(os.path.join(RES_DIR, "benchmark_metrics.json"), "w") as f:
    json.dump(comparison_metrics, f, indent=4)

print("\nBenchmark terminé. Résultats enregistrés dans results/benchmark_metrics.json")