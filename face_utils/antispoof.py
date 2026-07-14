"""Passive deep-learning anti-spoofing (Silent-Face-Anti-Spoofing).

Wraps Minivision's MiniFASNet models (the same ones used by
github.com/computervisioneng/face-attendance-system) to classify every
frame's face as REAL vs FAKE (printed photo / phone screen / replay).

IMPORTANT PROJECT NOTE: these are pretrained weights, used strictly for
spoof detection. Face *recognition* remains the from-scratch FaceNet —
no pretrained model touches identity decisions.

Unlike the reference repo we do NOT use its bundled Caffe face detector;
the face box comes from our own OpenCV Haar cascade, and both MiniFASNet
models are loaded once and cached (the reference reloads them from disk
every frame).
"""

import os
import sys
import threading

import numpy as np

import config

_lock = threading.Lock()
_models = None          # [(model, scale, w, h)] once loaded
_cropper = None
_to_tensor = None
_torch = None


def available():
    d = os.path.join(config.ANTISPOOF_DIR, "resources", "anti_spoof_models")
    return (config.ANTISPOOF_ENABLED and os.path.isdir(d)
            and any(f.endswith(".pth") for f in os.listdir(d)))


def _load():
    global _models, _cropper, _to_tensor, _torch
    if _models is not None:
        return _models
    with _lock:
        if _models is not None:
            return _models

        import torch
        import torch.nn.functional  # noqa: F401  (used via functional API)
        if config.ANTISPOOF_DIR not in sys.path:
            sys.path.insert(0, config.ANTISPOOF_DIR)
        from src.anti_spoof_predict import MODEL_MAPPING
        from src.generate_patches import CropImage
        from src.data_io import transform as trans
        from src.utility import get_kernel, parse_model_name

        device = torch.device("cpu")
        models_dir = os.path.join(config.ANTISPOOF_DIR,
                                  "resources", "anti_spoof_models")
        loaded = []
        for name in sorted(os.listdir(models_dir)):
            if not name.endswith(".pth"):
                continue
            h, w, model_type, scale = parse_model_name(name)
            model = MODEL_MAPPING[model_type](
                conv6_kernel=get_kernel(h, w)).to(device)
            state = torch.load(os.path.join(models_dir, name),
                               map_location=device)
            if next(iter(state)).startswith("module."):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state)
            model.eval()
            loaded.append((model, scale, w, h))

        _torch = torch
        _cropper = CropImage()
        _to_tensor = trans.Compose([trans.ToTensor()])
        _models = loaded
        return _models


def check(frame_bgr, bbox):
    """Classify the face in `bbox` (x, y, w, h from the Haar cascade).

    Returns (is_real: bool, real_prob: float). Each MiniFASNet votes with
    a softmax over [fake-2D, real, fake-3D]; votes are averaged exactly
    like the reference implementation.
    """
    models = _load()
    bbox = [int(v) for v in bbox]
    prediction = np.zeros((1, 3))
    for model, scale, w, h in models:
        img = _cropper.crop(org_img=frame_bgr, bbox=bbox, scale=scale,
                            out_w=w, out_h=h, crop=scale is not None)
        tensor = _to_tensor(img).unsqueeze(0)
        with _torch.no_grad():
            result = _torch.nn.functional.softmax(model(tensor), dim=1)
        prediction += result.cpu().numpy()

    prediction /= len(models)
    label = int(np.argmax(prediction))
    real_prob = float(prediction[0][1])
    is_real = (label == 1) and real_prob >= config.ANTISPOOF_REAL_THRESHOLD
    return is_real, real_prob
