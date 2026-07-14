"""Liveness detection (anti-spoofing) — hybrid active + passive.

A printed or on-screen photo passes plain face detection, so before any
face is trusted (registration capture AND attendance) the user must pass
a liveness session:

ACTIVE (challenge-response, the industry-recommended strongest method):
  1. Blink  — mandatory. Eyes are tracked with the Haar eye cascade;
     a real person produces an open -> closed -> open transition.
     A photo's eyes never close.
  2. One RANDOM extra challenge — turn head left / right (verified with
     the Haar *profile* cascade: a flat photo cannot produce a true
     profile view) or smile (Haar smile cascade). Randomization defeats
     pre-recorded video replays.

PASSIVE (screen-replay detection):
  Every frame's face region is checked for moiré interference patterns
  with an FFT peak analysis — LCD/phone screens photographed by a camera
  produce strong periodic frequency spikes that real skin does not.

Only OpenCV Haar cascades are used (the project's single allowed external
vision component). No pretrained anti-spoofing models.
"""

import random
import threading
import time

import cv2
import numpy as np

import config
from face_utils import antispoof
from face_utils import detector

_smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml")

CHALLENGE_TEXT = {
    "blink":      "Close your eyes for a moment, then open them",
    "turn_left":  "Slowly turn your head to the LEFT, then back",
    "turn_right": "Slowly turn your head to the RIGHT, then back",
    "smile":      "Give a big SMILE 😊",
}


# ---------------------------------------------------------------------------
# Per-frame observations
# ---------------------------------------------------------------------------

def _detect_profile(gray):
    """Returns 'left', 'right' or None using the Haar profile cascade."""
    min_size = config.RELAXED_MIN_FACE_SIZE // 2
    prof = detector._profile_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size))
    if len(prof):
        return "left"
    flipped = cv2.flip(gray, 1)
    prof = detector._profile_cascade.detectMultiScale(
        flipped, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size))
    if len(prof):
        return "right"
    return None


def _detect_smile(face_gray):
    """Smile in the lower half of the face. High minNeighbors keeps the
    smile cascade from firing on neutral mouths."""
    h = face_gray.shape[0]
    lower = face_gray[int(h * 0.5):]
    smiles = _smile_cascade.detectMultiScale(
        lower, scaleFactor=1.7, minNeighbors=22,
        minSize=(int(face_gray.shape[1] * 0.3), int(h * 0.15)))
    return len(smiles) > 0


def moire_score(face_gray):
    """FFT peak/median ratio outside the low-frequency core.

    Real faces have a smoothly decaying spectrum; screens photographed by
    a webcam show isolated periodic peaks (moiré / pixel-grid aliasing).
    """
    img = cv2.resize(face_gray, (128, 128)).astype(np.float32)
    img *= np.outer(np.hanning(128), np.hanning(128))  # window to kill edges
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img)))

    yy, xx = np.mgrid[-64:64, -64:64]
    r = np.sqrt(xx * xx + yy * yy)
    band = (r > 18) & (r < 60)           # ignore low-freq face content + corners
    values = spec[band]
    med = np.median(values)
    if med <= 0:
        return 0.0
    return float(values.max() / med)


def analyze_frame(frame_bgr):
    """One frame -> observation dict for the liveness state machine."""
    obs = {"face": False, "frontal": False, "profile": None,
           "eyes_open": None, "smile": False, "moire": 0.0, "fake": None}
    if frame_bgr is None or frame_bgr.size == 0:
        return obs
    if detector.frame_brightness(frame_bgr) < config.MIN_BRIGHTNESS:
        return obs

    frame_bgr = detector.enhance_frame(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = detector._clahe.apply(gray)

    faces = detector._boxes(detector._face_cascade, gray_eq, 5,
                            config.RELAXED_MIN_FACE_SIZE // 2)
    if len(faces) == 1:
        x, y, w, h = faces[0]
        face_gray = gray[y:y + h, x:x + w]
        obs["face"] = True
        obs["frontal"] = True
        obs["eyes_open"] = len(detector.detect_eyes(face_gray)) > 0
        obs["smile"] = _detect_smile(face_gray)
        obs["moire"] = moire_score(face_gray)
        if antispoof.available():
            is_real, _ = antispoof.check(frame_bgr, (x, y, w, h))
            obs["fake"] = not is_real
        return obs

    profile = _detect_profile(gray_eq)
    if profile:
        obs["face"] = True
        obs["profile"] = profile
    return obs


# ---------------------------------------------------------------------------
# Challenge state machine
# ---------------------------------------------------------------------------

class LivenessSession:
    """Random challenge sequence: blink + one extra, plus passive checks."""

    def __init__(self):
        self.challenges = ["blink",
                           random.choice(config.LIVENESS_EXTRA_CHALLENGES)]
        random.shuffle(self.challenges)
        self.idx = 0
        self.created = time.time()
        self.done = False
        self.failed = False
        self.fail_reason = ""
        self.moire_hits = 0
        self.fake_streak = 0
        self._reset_challenge_state()

    # -- public ------------------------------------------------------------

    @property
    def current(self):
        return None if self.done else self.challenges[self.idx]

    def status(self):
        return {
            "done": self.done,
            "failed": self.failed,
            "reason": self.fail_reason,
            "challenge": self.current,
            "instruction": CHALLENGE_TEXT.get(self.current, ""),
            "step": min(self.idx + 1, len(self.challenges)),
            "total": len(self.challenges),
        }

    def update(self, obs):
        """Feed one observation; returns status()."""
        if self.done or self.failed:
            return self.status()

        if time.time() - self.created > config.LIVENESS_TIMEOUT_S:
            self._fail("Liveness check timed out — please try again")
            return self.status()

        # passive screen detection accumulates over the whole session
        if obs["frontal"] and obs["moire"] > config.MOIRE_PEAK_RATIO:
            self.moire_hits += 1
            if self.moire_hits >= config.MOIRE_MAX_HITS:
                self._fail("Screen or printed photo suspected — "
                           "liveness check failed")
                return self.status()

        # deep-learning anti-spoofing: consecutive fake frames = spoof.
        # (A real person may get an occasional false 'fake'; a photo or
        # screen is flagged continuously.)
        if obs.get("fake") is True:
            self.fake_streak += 1
            if self.fake_streak >= config.ANTISPOOF_MAX_FAKE_STREAK:
                self._fail("Photo or screen detected by the anti-spoofing "
                           "model — liveness check failed")
                return self.status()
        elif obs.get("fake") is False:
            self.fake_streak = 0

        handler = getattr(self, "_step_" + self.current)
        if handler(obs):
            self.idx += 1
            self._reset_challenge_state()
            if self.idx >= len(self.challenges):
                self.done = True
        return self.status()

    # -- internals -----------------------------------------------------------

    def _fail(self, reason):
        self.failed = True
        self.fail_reason = reason

    def _reset_challenge_state(self):
        self.open_streak = 0
        self.closed_streak = 0
        self.blink_phase = "open"     # open -> closed -> reopen
        self.frontal_streak = 0
        self.profile_streak = 0
        self.smile_streak = 0
        self.face_lost = 0

    def _track_face_lost(self, obs):
        if not obs["face"]:
            self.face_lost += 1
            if self.face_lost > 12:   # ~2.5 s without any face: restart step
                self._reset_challenge_state()
        else:
            self.face_lost = 0

    def _step_blink(self, obs):
        self._track_face_lost(obs)
        if not obs["frontal"]:
            return False
        if obs["eyes_open"]:
            if self.blink_phase == "open":
                self.open_streak += 1
            elif self.blink_phase == "closed":
                # eyes reopened: valid blink only if closure was long enough
                if (config.BLINK_MIN_CLOSED_FRAMES <= self.closed_streak
                        <= config.BLINK_MAX_CLOSED_FRAMES):
                    return True
                self.blink_phase = "open"
                self.open_streak = 1
                self.closed_streak = 0
        else:
            if self.blink_phase == "open" and self.open_streak >= 3:
                self.blink_phase = "closed"
                self.closed_streak = 1
            elif self.blink_phase == "closed":
                self.closed_streak += 1
                if self.closed_streak > config.BLINK_MAX_CLOSED_FRAMES:
                    self.blink_phase = "open"
                    self.open_streak = 0
                    self.closed_streak = 0
        return False

    def _step_turn(self, obs):
        """A real 3D head turn: frontal baseline, then a true profile view
        (direction-agnostic — tilting a flat photo cannot make a profile)."""
        self._track_face_lost(obs)
        if obs["frontal"]:
            self.frontal_streak += 1
        if self.frontal_streak >= 2 and obs["profile"]:
            self.profile_streak += 1
            if self.profile_streak >= 2:
                return True
        return False

    _step_turn_left = _step_turn
    _step_turn_right = _step_turn

    def _step_smile(self, obs):
        self._track_face_lost(obs)
        if not obs["frontal"]:
            return False
        if obs["smile"]:
            self.smile_streak += 1
            if self.smile_streak >= 2:
                return True
        else:
            self.smile_streak = 0
        return False


# ---------------------------------------------------------------------------
# Session manager (per student + purpose)
# ---------------------------------------------------------------------------

_sessions = {}
_lock = threading.Lock()


def start_session(key):
    with _lock:
        session = LivenessSession()
        _sessions[key] = session
        return session.status()


def feed_frame(key, frame_bgr):
    with _lock:
        session = _sessions.get(key)
    if session is None:
        status = start_session(key)
        status["restarted"] = True
        return status
    obs = analyze_frame(frame_bgr)
    with _lock:
        status = session.update(obs)
        if status["failed"]:
            _sessions.pop(key, None)   # a fresh session on next attempt
    return status


def clear_session(key):
    with _lock:
        _sessions.pop(key, None)
