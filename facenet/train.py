"""Training pipeline: FaceNet from scratch with triplet loss.

Runs in a background thread so the Flask UI can poll live progress
(epoch, loss, accuracy, val accuracy, progress %, ETA).
"""

import os
import threading
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras

import config
import database
from facenet import data as data_lib
from facenet import model as model_lib
from facenet import triplet

# ---------------------------------------------------------------------------
# Shared training state (polled by the web UI)
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
training_state = {
    "running": False,
    "stage": "idle",        # idle | preparing | training | embedding | done | error
    "message": "",
    "epoch": 0,
    "total_epochs": config.EPOCHS,
    "loss": None,
    "accuracy": None,
    "val_loss": None,
    "val_accuracy": None,
    "progress": 0.0,        # 0..100
    "eta_seconds": None,
    "history": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []},
}


def _update(**kwargs):
    with _state_lock:
        training_state.update(kwargs)


def get_state():
    with _state_lock:
        return dict(training_state)


class ProgressCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_times.append(time.time() - self._epoch_start)
        avg = float(np.mean(self.epoch_times[-5:]))
        remaining = max(0, config.EPOCHS - (epoch + 1))
        with _state_lock:
            h = training_state["history"]
            h["loss"].append(float(logs.get("loss", 0)))
            h["accuracy"].append(float(logs.get("triplet_accuracy", 0)))
            h["val_loss"].append(float(logs.get("val_loss", 0)))
            h["val_accuracy"].append(float(logs.get("val_triplet_accuracy", 0)))
        _update(
            epoch=epoch + 1,
            loss=round(float(logs.get("loss", 0)), 4),
            accuracy=round(float(logs.get("triplet_accuracy", 0)), 4),
            val_loss=round(float(logs.get("val_loss", 0)), 4),
            val_accuracy=round(float(logs.get("val_triplet_accuracy", 0)), 4),
            progress=round(100.0 * (epoch + 1) / config.EPOCHS, 1),
            eta_seconds=int(avg * remaining),
        )


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_title("Triplet Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Val accuracy")
    axes[1].set_title("Triplet Accuracy (rank-1 in batch)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("FaceNet from scratch — training metrics")
    fig.tight_layout()
    fig.savefig(config.HISTORY_PLOT_PATH, dpi=120)
    plt.close(fig)


def generate_all_embeddings(base_model=None):
    """Embed every student's dataset with the trained model and store the
    mean (re-normalized) 128-D vector per student in the database."""
    if base_model is None:
        base_model = keras.models.load_model(config.MODEL_PATH, compile=False)

    from facenet.recognize import embed_images  # local import, avoids cycle

    for student_id in data_lib.list_student_dirs():
        if not database.find_student(student_id=student_id):
            continue  # stray folder without a registered student
        folder = os.path.join(config.DATASET_DIR, student_id)
        import cv2
        imgs = []
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        if not imgs:
            continue
        embs = embed_images(imgs, model=base_model)
        mean = embs.mean(axis=0)
        mean /= max(np.linalg.norm(mean), 1e-10)
        database.save_embedding(student_id, mean)
        database.update_student_status(student_id, "completed")


def _train():
    try:
        _update(running=True, stage="preparing",
                message="Loading dataset...", epoch=0, progress=0.0,
                loss=None, accuracy=None, val_loss=None, val_accuracy=None,
                eta_seconds=None,
                history={"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []})

        images, labels, id_to_label = data_lib.load_dataset()
        n_classes = len(id_to_label)
        if n_classes < 2:
            _update(running=False, stage="error",
                    message="Triplet loss needs at least 2 registered students "
                            f"with face datasets (found {n_classes}).")
            return

        train_idx, val_idx = data_lib.split_train_val(labels)
        train_seq = data_lib.PKBatchSequence(
            images[train_idx], labels[train_idx],
            steps=config.STEPS_PER_EPOCH, shuffle=True)
        val_seq = data_lib.PKBatchSequence(
            images[val_idx], labels[val_idx],
            steps=max(8, config.STEPS_PER_EPOCH // 5), shuffle=False)

        _update(message=f"Training on {len(train_idx)} images of "
                        f"{n_classes} students (val: {len(val_idx)} images)...",
                stage="training")

        training_model, base_model = model_lib.build_training_model()
        training_model.compile(
            optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
            loss=triplet.triplet_loss,
            metrics=[triplet.triplet_accuracy],
        )

        ckpt_path = os.path.join(config.MODELS_DIR, "best.weights.h5")
        callbacks = [
            ProgressCallback(),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                ckpt_path, monitor="val_loss", save_best_only=True,
                save_weights_only=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE, min_lr=config.MIN_LR),
        ]

        training_model.fit(
            train_seq, validation_data=val_seq,
            epochs=config.EPOCHS, callbacks=callbacks, verbose=2)

        # EarlyStopping already restored the best weights; save the inner
        # embedding network only (augmentation layers are training-only).
        base_model.save(config.MODEL_PATH)

        with _state_lock:
            history = {k: list(v) for k, v in training_state["history"].items()}
        if history["loss"]:
            _plot_history(history)

        _update(stage="embedding",
                message="Generating 128-D embeddings for all students...")
        generate_all_embeddings(base_model)

        # reload recognition model cache
        from facenet import recognize
        recognize.reload_model()

        _update(running=False, stage="done", progress=100.0,
                message="Training complete. Model saved and embeddings stored.")
    except Exception as exc:  # surface errors to the UI
        traceback.print_exc()
        _update(running=False, stage="error", message=f"Training failed: {exc}")


def start_training():
    """Kick off training in a background thread. Returns (ok, message)."""
    with _state_lock:
        if training_state["running"]:
            return False, "Training is already running."
        training_state["running"] = True
        training_state["stage"] = "preparing"
    threading.Thread(target=_train, daemon=True).start()
    return True, "Training started."
