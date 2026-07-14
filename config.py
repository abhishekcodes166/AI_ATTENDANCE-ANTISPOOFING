"""Central configuration for the AI Attendance System."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASE_PATH = os.path.join(BASE_DIR, "attendance.db")
MODEL_PATH = os.path.join(MODELS_DIR, "facenet_model.keras")
HISTORY_PLOT_PATH = os.path.join(MODELS_DIR, "training_history.png")

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = "change-this-secret-key-in-production"

# Admin credentials (change before deploying anywhere real)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ---------------------------------------------------------------------------
# Face detection / capture
# ---------------------------------------------------------------------------
FACE_IMAGE_SIZE = 96          # model input size (96x96x3)
SAVED_IMAGE_SIZE = 160        # aligned crops stored on disk at this size

MIN_FACE_SIZE = 100           # min face box width/height in the 640x480 frame
BLUR_THRESHOLD = 45.0         # variance of Laplacian; below this = blurry
CENTER_TOLERANCE = 0.24       # face center must be within this fraction of frame center
DUPLICATE_DIFF_THRESHOLD = 4.0  # mean abs pixel diff vs previous accepted image
MIN_BRIGHTNESS = 42.0         # mean gray level below this = "too dark"

# Guided capture plan: (pose key, on-screen instruction, images, eyes required)
# For turned/tilted poses the detector automatically relaxes its gates and
# also tries the Haar *profile* cascade, so students don't get stuck.
CAPTURE_POSES = [
    ("straight", "Look straight at the camera", 40, True),
    ("left",     "Turn your head slightly LEFT", 30, False),
    ("right",    "Turn your head slightly RIGHT", 30, False),
    ("up",       "Tilt your chin UP a little",  20, False),
    ("down",     "Tilt your chin DOWN a little", 20, False),
    ("smile",    "Smile at the camera",         30, True),
    ("neutral",  "Relax — neutral expression",  30, True),
]
# per-pose relaxation of the quality gates
RELAXED_POSES = {"left", "right", "up", "down"}
RELAXED_BLUR_FACTOR = 0.65     # blur threshold multiplier for relaxed poses
RELAXED_CENTER_TOLERANCE = 0.32
RELAXED_MIN_FACE_SIZE = 85
TOTAL_CAPTURE_IMAGES = sum(p[2] for p in CAPTURE_POSES)  # = 200
CAPTURE_INTERVAL_MS = 400     # browser sends one frame every 0.4 s

# ---------------------------------------------------------------------------
# FaceNet training (from scratch — no pretrained weights anywhere)
# ---------------------------------------------------------------------------
EMBEDDING_SIZE = 128
TRIPLET_MARGIN = 0.2
TRIPLET_MINING = "semi-hard"   # "semi-hard" or "hard"

CLASSES_PER_BATCH = 8          # P identities per batch (capped at #students)
IMAGES_PER_CLASS = 4           # K images per identity per batch
EPOCHS = 60
STEPS_PER_EPOCH = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-5

# ---------------------------------------------------------------------------
# Liveness / anti-spoofing (challenge-response + passive screen detection)
# ---------------------------------------------------------------------------
LIVENESS_ENABLED = True
LIVENESS_FRAME_INTERVAL_MS = 200   # browser sends frames faster during checks
LIVENESS_TIMEOUT_S = 40            # whole challenge sequence must finish in this
LIVENESS_TTL_CAPTURE_S = 300       # passed check authorizes capture this long
LIVENESS_TTL_ATTENDANCE_S = 90     # passed check authorizes recognition this long
LIVENESS_EXTRA_CHALLENGES = ["turn_left", "turn_right", "smile"]  # one at random
# blink timing at LIVENESS_FRAME_INTERVAL_MS sampling
BLINK_MIN_CLOSED_FRAMES = 2        # eyes must stay closed >= this many frames
BLINK_MAX_CLOSED_FRAMES = 10       # longer = looked away / covered, reset
# passive moiré (screen replay) detection via FFT peak analysis
MOIRE_PEAK_RATIO = 55.0            # spectral peak/median ratio flagged as screen
MOIRE_MAX_HITS = 5                 # this many flagged frames fails the session

# Deep-learning anti-spoofing (Silent-Face-Anti-Spoofing / MiniFASNet).
# NOTE: these are PRETRAINED weights — used ONLY for spoof detection, never
# for face recognition, which remains a from-scratch FaceNet.
ANTISPOOF_ENABLED = True
ANTISPOOF_DIR = os.path.join(BASE_DIR, "third_party", "Silent-Face-Anti-Spoofing")
ANTISPOOF_REAL_THRESHOLD = 0.50    # averaged real-class probability required
ANTISPOOF_MAX_FAKE_STREAK = 6      # consecutive fake frames fail a liveness session

# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD = 0.65   # cosine similarity threshold
REQUIRE_LOGIN_MATCH = True     # recognized face must match the logged-in student

# ---------------------------------------------------------------------------
# Defaults seeded into the database on first run; after that the admin
# manages subjects and semesters from the admin dashboard.
# ---------------------------------------------------------------------------
DEFAULT_SEMESTERS = ["1", "2", "3", "4", "5", "6", "7", "8"]

DEFAULT_SUBJECTS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Data Structures",
    "Operating Systems",
    "Computer Networks",
    "Database Management Systems",
]

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
