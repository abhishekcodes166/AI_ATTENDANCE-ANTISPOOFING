"""Dataset loading and balanced P x K batch sampling for triplet training.

Triplet loss needs every batch to contain several images of several
identities, so batches are built as P identities x K images (a "PK batch")
instead of uniform random sampling.
"""

import os
import random

import cv2
import numpy as np
import keras

import config


def list_student_dirs():
    """Student IDs that have a dataset folder with images."""
    if not os.path.isdir(config.DATASET_DIR):
        return []
    out = []
    for name in sorted(os.listdir(config.DATASET_DIR)):
        path = os.path.join(config.DATASET_DIR, name)
        if os.path.isdir(path) and any(
                f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(path)):
            out.append(name)
    return out


def load_dataset():
    """Load every collected face image into memory.

    Returns (images uint8 [N,S,S,3], integer labels [N], id_to_label dict).
    Images are stored on disk as aligned RGB crops; here they are resized
    to the model input size.
    """
    size = config.FACE_IMAGE_SIZE
    images, labels, id_to_label = [], [], {}

    for student_id in list_student_dirs():
        folder = os.path.join(config.DATASET_DIR, student_id)
        label = id_to_label.setdefault(student_id, len(id_to_label))
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            images.append(img)
            labels.append(label)

    if not images:
        return (np.zeros((0, size, size, 3), np.uint8),
                np.zeros((0,), np.int32), {})
    return np.stack(images), np.asarray(labels, np.int32), id_to_label


def split_train_val(labels, val_fraction=None, seed=42):
    """Per-identity train/validation split so every student appears in both."""
    val_fraction = val_fraction or config.VALIDATION_SPLIT
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        idx = list(np.where(labels == label)[0])
        rng.shuffle(idx)
        n_val = max(2, int(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.asarray(train_idx), np.asarray(val_idx)


class PKBatchSequence(keras.utils.PyDataset):
    """Yields balanced batches of P identities x K images for online mining."""

    def __init__(self, images, labels, steps, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.labels = labels
        self.steps = steps
        self.shuffle = shuffle
        self.rng = random.Random(1234 if not shuffle else None)

        self.by_class = {
            int(c): list(np.where(labels == c)[0]) for c in np.unique(labels)
        }
        self.classes = list(self.by_class.keys())
        self.P = min(config.CLASSES_PER_BATCH, len(self.classes))
        self.K = config.IMAGES_PER_CLASS

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        chosen = self.rng.sample(self.classes, self.P)
        idx = []
        for c in chosen:
            pool = self.by_class[c]
            if len(pool) >= self.K:
                idx.extend(self.rng.sample(pool, self.K))
            else:
                idx.extend(self.rng.choices(pool, k=self.K))
        batch_x = self.images[idx].astype(np.float32) / 255.0
        batch_y = self.labels[idx].astype(np.int32)
        return batch_x, batch_y
