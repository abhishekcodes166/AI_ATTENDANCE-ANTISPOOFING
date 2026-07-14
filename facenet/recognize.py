"""Recognition: embed a face with the trained FaceNet and match it against
stored student embeddings using cosine similarity.

No softmax classifier anywhere — recognition is pure embedding matching,
as in real-world face recognition systems.
"""

import os
import threading

import cv2
import numpy as np

import config
import database

_model = None
_model_lock = threading.Lock()


def model_exists():
    return os.path.exists(config.MODEL_PATH)


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            if not model_exists():
                return None
            import keras
            _model = keras.models.load_model(config.MODEL_PATH, compile=False)
        return _model


def reload_model():
    global _model
    with _model_lock:
        _model = None


def _preprocess(images_rgb):
    size = config.FACE_IMAGE_SIZE
    batch = np.stack([
        cv2.resize(img, (size, size)) for img in images_rgb
    ]).astype(np.float32) / 255.0
    return batch


def embed_images(images_rgb, model=None):
    """RGB face crops -> L2-normalized 128-D embeddings [N, 128]."""
    model = model or _get_model()
    if model is None:
        raise RuntimeError("FaceNet model has not been trained yet.")
    batch = _preprocess(images_rgb)
    embs = model.predict(batch, verbose=0, batch_size=64)
    # model already L2-normalizes; renormalize defensively
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-10)


def embed_face(face_rgb):
    return embed_images([face_rgb])[0]


def match_embedding(embedding):
    """Cosine-similarity match against every stored student embedding.

    Returns (student_id or None, best_similarity). Embeddings are unit
    vectors, so cosine similarity is just the dot product."""
    stored = database.load_all_embeddings()
    if not stored:
        return None, 0.0

    ids = list(stored.keys())
    matrix = np.stack([stored[i] for i in ids])          # [M, 128]
    sims = matrix @ embedding                            # [M]
    best = int(np.argmax(sims))
    best_sim = float(sims[best])

    if best_sim >= config.RECOGNITION_THRESHOLD:
        return ids[best], best_sim
    return None, best_sim
