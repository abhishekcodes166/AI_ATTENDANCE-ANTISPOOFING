"""Face detection, alignment and image-quality checks.

The ONLY external vision components in this project are OpenCV Haar
cascades (frontal face, profile face, eyes) — allowed by the project
constraints. Everything downstream — embedding and recognition — is a
FaceNet model trained from scratch.

Robustness features:
  * Illumination enhancement (gamma correction + CLAHE) so faces are
    detected in dim rooms.
  * Pose-aware detection: for "turn left/right" capture steps the frontal
    cascade is tried first, then the Haar *profile* cascade (and its
    mirror), with relaxed quality gates — so students are not rejected
    for doing exactly what the instruction asked.
"""

import cv2
import numpy as np

import config

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# alt2 is noticeably more tolerant of pitch (chin up/down) than the default
_face_cascade_alt2 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml")
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


# ---------------------------------------------------------------------------
# Illumination
# ---------------------------------------------------------------------------

def frame_brightness(frame_bgr):
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())


def enhance_frame(frame_bgr):
    """Gamma-correct dim frames so detection and embeddings still work in
    poor lighting. Applied identically during capture AND recognition so
    the model always sees consistently-lit faces."""
    mean = frame_brightness(frame_bgr)
    if mean < 95:
        # pick gamma that maps the current mean toward mid-gray:
        # (mean/255)^(1/gamma) = 0.5  =>  gamma = log(mean/255)/log(0.5)
        gamma = np.log(max(mean, 10.0) / 255.0) / np.log(0.5)
        gamma = float(np.clip(gamma, 1.0, 2.2))
        table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
        frame_bgr = cv2.LUT(frame_bgr, table)
    return frame_bgr


def _detection_gray(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _clahe.apply(gray)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _boxes(cascade, gray, min_neighbors, min_size):
    found = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=min_neighbors,
        minSize=(min_size, min_size))
    return list(found)


def detect_faces(frame_bgr, allow_profile=False, relaxed=False, tilt_ok=False):
    """Return list of (x, y, w, h) face boxes.

    allow_profile -- also try the Haar profile cascade (for turned heads).
    relaxed       -- lower detector strictness (for tilted/turned poses).
    tilt_ok       -- chin up/down poses: fall back to the alt2 frontal
                     cascade and very permissive settings (the default
                     frontal cascade handles yaw fine but is weak on pitch).
    """
    gray = _detection_gray(frame_bgr)
    min_size = (config.RELAXED_MIN_FACE_SIZE if relaxed
                else config.MIN_FACE_SIZE) // 2
    neighbors = 4 if relaxed else 5

    faces = _boxes(_face_cascade, gray, neighbors, min_size)
    if faces:
        return faces

    if relaxed:  # frontal cascade, more permissive
        faces = _boxes(_face_cascade, gray, 3, min_size)
        if faces:
            return faces

    if tilt_ok:  # pitch-tolerant fallbacks for chin up/down
        small = max(20, int(min_size * 0.8))
        faces = _boxes(_face_cascade_alt2, gray, 3, small)
        if faces:
            return faces
        faces = _boxes(_face_cascade_alt2, gray, 2, small)
        if faces:
            return faces
        faces = _boxes(_face_cascade, gray, 2, small)
        if faces:
            return faces
        # a pitched face is vertically foreshortened; stretching the frame
        # back toward normal proportions lets the cascade catch strong tilts
        for stretch in (1.35, 1.6):
            tall = cv2.resize(gray, (gray.shape[1], int(gray.shape[0] * stretch)))
            faces = (_boxes(_face_cascade_alt2, tall, 3, small)
                     or _boxes(_face_cascade, tall, 3, small))
            if faces:
                return [(x, int(y / stretch), w, max(1, int(h / stretch)))
                        for (x, y, w, h) in faces]

    if allow_profile:
        # profile cascade detects LEFT-facing profiles; mirror for right
        faces = _boxes(_profile_cascade, gray, 4, min_size)
        if faces:
            return faces
        flipped = cv2.flip(gray, 1)
        faces = _boxes(_profile_cascade, flipped, 4, min_size)
        if faces:
            w_frame = gray.shape[1]
            return [(w_frame - x - w, y, w, h) for (x, y, w, h) in faces]
    return []


def detect_eyes(face_gray):
    """Detect eyes inside a grayscale face crop. Returns list of boxes."""
    h = face_gray.shape[0]
    upper = face_gray[: int(h * 0.65)]  # eyes live in the upper face
    eyes = _eye_cascade.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=4)
    return list(eyes)


def align_face(frame_bgr, box):
    """Crop the face; if both eyes are found, rotate so the eye line is
    horizontal before cropping. Returns an RGB crop at SAVED_IMAGE_SIZE."""
    x, y, w, h = box
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    eyes = detect_eyes(gray[y:y + h, x:x + w])

    if len(eyes) >= 2:
        # take the two largest detections, order left->right
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eyes = sorted(eyes, key=lambda e: e[0])
        (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
        left = (x + ex1 + ew1 / 2, y + ey1 + eh1 / 2)
        right = (x + ex2 + ew2 / 2, y + ey2 + eh2 / 2)
        angle = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
        if abs(angle) < 25:  # sanity: reject wild eye detections
            center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            frame_bgr = cv2.warpAffine(
                frame_bgr, rot, (frame_bgr.shape[1], frame_bgr.shape[0]),
                flags=cv2.INTER_LINEAR)

    # generous margin around the box, clipped to the frame
    mx, my = int(w * 0.12), int(h * 0.12)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame_bgr.shape[1], x + w + mx), min(frame_bgr.shape[0], y + h + my)
    crop = frame_bgr[y0:y1, x0:x1]
    crop = cv2.resize(crop, (config.SAVED_IMAGE_SIZE, config.SAVED_IMAGE_SIZE))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def sharpness(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ---------------------------------------------------------------------------
# Capture quality gates
# ---------------------------------------------------------------------------

def check_frame(frame_bgr, prev_fingerprint=None, pose="straight",
                require_eyes=True):
    """Run every capture-quality gate on a webcam frame.

    Gates are pose-aware: turned/tilted poses get relaxed thresholds and
    profile-cascade fallback so the guided capture never fights its own
    instructions.

    Returns (ok, reason, face_crop_rgb, fingerprint).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False, "Empty frame", None, None

    if frame_brightness(frame_bgr) < config.MIN_BRIGHTNESS:
        return False, "Too dark — turn on more light or face a window", None, None

    frame_bgr = enhance_frame(frame_bgr)

    relaxed = pose in config.RELAXED_POSES
    profile_ok = pose in ("left", "right")
    tilt_ok = pose in ("up", "down")
    min_face = config.RELAXED_MIN_FACE_SIZE if relaxed else config.MIN_FACE_SIZE
    if tilt_ok:
        min_face = int(min_face * 0.8)   # tilted faces box smaller
    center_tol = (config.RELAXED_CENTER_TOLERANCE if relaxed
                  else config.CENTER_TOLERANCE)
    blur_thr = config.BLUR_THRESHOLD * (
        config.RELAXED_BLUR_FACTOR if relaxed else 1.0)

    faces = detect_faces(frame_bgr, allow_profile=profile_ok,
                         relaxed=relaxed, tilt_ok=tilt_ok)
    if len(faces) == 0:
        return False, "No face detected — face the camera", None, None
    if len(faces) > 1:
        # The permissive tilt pass can produce spurious extra boxes; keep the
        # largest unless a second detection is comparably big (real 2nd person)
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        if not tilt_ok or faces[1][2] * faces[1][3] > 0.6 * faces[0][2] * faces[0][3]:
            return False, "Multiple faces detected — only one person allowed", None, None

    x, y, w, h = faces[0]
    fh, fw = frame_bgr.shape[:2]

    if w < min_face or h < min_face:
        return False, "Move a little closer to the camera", None, None

    # face must be roughly centered
    cx, cy = x + w / 2, y + h / 2
    if (abs(cx - fw / 2) > fw * center_tol
            or abs(cy - fh / 2) > fh * center_tol):
        return False, "Keep your face inside the frame", None, None

    face_bgr = frame_bgr[y:y + h, x:x + w]
    if sharpness(face_bgr) < blur_thr:
        return False, "Hold still — image is blurry", None, None

    if require_eyes and not relaxed:
        gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        if len(detect_eyes(gray_face)) == 0:
            return False, "Eyes not visible", None, None

    # anti-spoofing: never save a frame that looks like a photo/screen
    from face_utils import antispoof
    if antispoof.available():
        is_real, _ = antispoof.check(frame_bgr, (x, y, w, h))
        if not is_real:
            return False, "Photo/screen detected — real faces only", None, None

    fingerprint = cv2.resize(
        cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
    if prev_fingerprint is not None:
        diff = float(np.mean(np.abs(fingerprint - prev_fingerprint)))
        if diff < config.DUPLICATE_DIFF_THRESHOLD:
            return False, "Move slightly — frame too similar to the last one", None, None

    crop_rgb = align_face(frame_bgr, (x, y, w, h))
    return True, "Captured", crop_rgb, fingerprint


def extract_face_for_recognition(frame_bgr):
    """Detect + align a single face for attendance recognition.

    Returns (crop_rgb, box) or (None, reason)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None, "Empty frame"
    if frame_brightness(frame_bgr) < config.MIN_BRIGHTNESS:
        return None, "Too dark — turn on more light"

    frame_bgr = enhance_frame(frame_bgr)
    faces = detect_faces(frame_bgr, relaxed=True)
    if len(faces) == 0:
        return None, "No face detected"
    if len(faces) > 1:
        return None, "Multiple faces detected"
    box = faces[0]
    if sharpness(frame_bgr[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]) \
            < config.BLUR_THRESHOLD * 0.5:
        return None, "Hold still — image is blurry"
    return align_face(frame_bgr, tuple(box)), tuple(int(v) for v in box)
