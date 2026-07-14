# AI Face Recognition Attendance System

A research-style face recognition attendance system. The **FaceNet architecture is
implemented and trained completely from scratch** — random initialization, no
pretrained weights, no external face recognition libraries. The only external
vision component is the **OpenCV Haar Cascade** used for face detection (allowed
by the project constraints).

## Pipeline

```
Student Registration
→ Automatic Face Dataset Collection (~200 guided images)
→ OpenCV Haar Cascade Face Detection
→ Face Alignment (eye-line rotation) & Preprocessing
→ FaceNet (random init, BN + ReLU) → 128-D Embedding → L2 Normalization
→ Triplet Loss (manual TF implementation, online semi-hard mining)
→ Embedding Database (SQLite)
→ Live Webcam → Detection → Embedding → Cosine Similarity → Attendance
```

## Setup

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python app.py
```

Open http://127.0.0.1:5001

Admin panel: http://127.0.0.1:5001/admin — username `admin`, password `admin123`
(change in `config.py`).

## Workflow

The landing page is the student login; new students register once from there.

1. **Register (one-time)** — student creates an account (unique Student ID + Email).
2. **Automatic capture** — the webcam collects ~200 face images with on-screen
   pose guidance (straight / left / right / up / down / smile / neutral).
   Frames are quality-gated: single face, centered, sharp, eyes visible,
   not a duplicate of the previous frame. Saved to `dataset/<student_id>/NNN.jpg`.
   Turned/tilted poses automatically use relaxed gates plus the Haar *profile*
   cascade, and dim frames are gamma-corrected (CLAHE-assisted detection), so
   capture keeps working in imperfect lighting.
3. **Admin trains the model** — FaceNet trained from scratch with triplet loss
   (semi-hard mining), data augmentation, EarlyStopping, ModelCheckpoint,
   ReduceLROnPlateau, validation split. Live epoch/loss/accuracy/ETA in the
   admin UI. Saves `models/facenet_model.keras` + a metrics plot, then stores
   each student's mean 128-D embedding in SQLite (`embeddings` table).
4. **Student login** — email + password. No face registration ever again.
5. **Mark attendance** — Haar detection → alignment → FaceNet embedding →
   cosine similarity against all stored embeddings. Above threshold ⇒
   attendance marked (once per subject per day); otherwise **Unknown Person**.
6. **New student later** — register → capture → admin retrains → embeddings
   regenerated. Existing students are unaffected.
7. **Subjects** — managed by the admin from the dashboard (add/remove);
   defaults are seeded from `config.DEFAULT_SUBJECTS` on first run.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask app — all routes and the capture/attendance APIs |
| `config.py` | Every tunable: thresholds, capture plan, training hyperparameters |
| `database.py` | SQLite layer: students, embeddings, attendance |
| `face_utils/detector.py` | Haar cascade detection, eye-based alignment, quality gates |
| `facenet/model.py` | FaceNet architecture (from scratch) |
| `facenet/triplet.py` | Manual triplet loss + semi-hard / batch-hard online mining |
| `facenet/data.py` | Dataset loading + balanced P×K batch sampling |
| `facenet/train.py` | Training pipeline, live progress state, metric plots, enrollment |
| `facenet/recognize.py` | Embedding generation + cosine-similarity matching |
| `dataset/` | Collected face crops per student |
| `models/` | `facenet_model.keras`, training plot |
| `attendance.db` | SQLite database (students / embeddings / attendance) |

## Liveness / Anti-Spoofing

A photo or phone screen cannot register or mark attendance. Two layers run
before any face is trusted, at BOTH registration capture and attendance:

1. **Active challenge-response** — a mandatory blink plus one random
   challenge (turn left / turn right / smile), verified with OpenCV Haar
   cascades (eye, profile, smile). A static photo can't blink and a flat
   photo can't produce a true profile view; randomization defeats
   pre-recorded replays. An FFT moiré check also flags screen patterns.
2. **MiniFASNet deep anti-spoofing** (`third_party/Silent-Face-Anti-Spoofing`,
   as used by computervisioneng/face-attendance-system) — every frame's
   face is classified real-vs-fake; spoofed frames are rejected during
   capture, recognition, and liveness. ⚠️ These are *pretrained* weights,
   used ONLY for spoof detection — face recognition itself remains the
   from-scratch FaceNet.

Tuning lives in `config.py` (`LIVENESS_*`, `ANTISPOOF_*`, `MOIRE_*`).

## Notes

- Recognition threshold, triplet margin, batch shape (P×K), epochs etc. are in
  `config.py`. `RECOGNITION_THRESHOLD = 0.65` cosine similarity by default.
- Training needs **at least 2 registered students** with datasets (triplet loss
  requires negatives).
- No softmax classifier is used anywhere — recognition is pure embedding
  matching, so new students only require retraining the metric space, not a
  classification head.
