"""AI Attendance System — Flask application.

Workflow:
  1. One-time student registration (unique Student ID + Email)
  2. Automatic guided face dataset collection (~200 images via webcam)
  3. Admin trains FaceNet from scratch (triplet loss, no pretrained weights)
  4. Student logs in with email/password
  5. Attendance marked via live face recognition (cosine similarity)
"""

import base64
import csv
import functools
import io
import os
import shutil
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import config
import database
from face_utils import detector
from face_utils import liveness
from facenet import recognize
from facenet import train as trainer

app = Flask(__name__)
app.secret_key = config.SECRET_KEY


# ---------------------------------------------------------------------------
# Capture sessions (server-side state for guided dataset collection)
# ---------------------------------------------------------------------------
_capture_lock = threading.Lock()
_capture_sessions = {}  # student_id -> {"count", "pose_index", "pose_count", "fingerprint"}


def _decode_frame(data_url):
    """base64 data-URL (from the browser canvas) -> BGR frame."""
    try:
        payload = data_url.split(",", 1)[1]
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _pose_info(pose_index, pose_count):
    key, instruction, target, eyes = config.CAPTURE_POSES[pose_index]
    return {"key": key, "instruction": instruction,
            "captured": pose_count, "target": target,
            "step": pose_index + 1, "steps": len(config.CAPTURE_POSES)}


# ---------------------------------------------------------------------------
# Liveness helpers
# ---------------------------------------------------------------------------

def _liveness_key(purpose):
    return f"{session['student_id']}:{purpose}"


def _liveness_valid(purpose):
    """Has the logged-in student passed a liveness check recently?"""
    if not config.LIVENESS_ENABLED:
        return True
    passed_at = session.get(f"liveness_{purpose}")
    if not passed_at:
        return False
    ttl = (config.LIVENESS_TTL_CAPTURE_S if purpose == "capture"
           else config.LIVENESS_TTL_ATTENDANCE_S)
    return (datetime.now().timestamp() - passed_at) <= ttl


def _liveness_consume(purpose):
    session.pop(f"liveness_{purpose}", None)
    liveness.clear_session(_liveness_key(purpose))


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def student_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if "student_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return view(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing = login/registration. Authenticated users go straight to
    their own dashboard."""
    if session.get("student_id"):
        return redirect(url_for("dashboard"))
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    semesters = database.all_semesters()
    if request.method == "GET":
        return render_template("register.html", error=None, form={},
                               semesters=semesters)

    form = {k: request.form.get(k, "").strip() for k in
            ("student_id", "full_name", "roll_number", "department",
             "semester", "section", "email")}
    password = request.form.get("password", "")

    if not all(form.values()) or not password:
        return render_template("register.html",
                               error="All fields are required.", form=form,
                               semesters=semesters)

    if form["semester"] not in semesters:
        return render_template("register.html",
                               error="Please choose a valid semester.",
                               form=form, semesters=semesters)

    # Uniqueness check: a student registers only ONCE.
    if database.find_student(student_id=form["student_id"],
                             email=form["email"]):
        return render_template(
            "register.html",
            error="This student is already registered. Please log in.",
            form=form, semesters=semesters)

    database.create_student(
        **form, password_hash=generate_password_hash(password))

    session.clear()
    session["student_id"] = form["student_id"]
    session["student_name"] = form["full_name"]
    return redirect(url_for("capture"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    student = database.find_student(email=email)
    if not student or not check_password_hash(student["password_hash"], password):
        return render_template("login.html", error="Invalid email or password.")

    session.clear()
    session["student_id"] = student["student_id"]
    session["student_name"] = student["full_name"]
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Liveness API (anti-spoofing challenges before capture / attendance)
# ---------------------------------------------------------------------------

@app.route("/api/liveness/start", methods=["POST"])
@student_required
def api_liveness_start():
    purpose = request.json.get("purpose", "")
    if purpose not in ("capture", "attendance"):
        return jsonify({"error": "bad purpose"}), 400
    session.pop(f"liveness_{purpose}", None)
    status = liveness.start_session(_liveness_key(purpose))
    status["interval_ms"] = config.LIVENESS_FRAME_INTERVAL_MS
    return jsonify(status)


@app.route("/api/liveness/frame", methods=["POST"])
@student_required
def api_liveness_frame():
    purpose = request.json.get("purpose", "")
    if purpose not in ("capture", "attendance"):
        return jsonify({"error": "bad purpose"}), 400
    frame = _decode_frame(request.json.get("image", ""))
    status = liveness.feed_frame(_liveness_key(purpose), frame)
    if status["done"]:
        session[f"liveness_{purpose}"] = datetime.now().timestamp()
    return jsonify(status)


# ---------------------------------------------------------------------------
# Face dataset collection (one-time, right after registration)
# ---------------------------------------------------------------------------

@app.route("/capture")
@student_required
def capture():
    student = database.find_student(student_id=session["student_id"])
    if student["registration_status"] != "registered":
        return redirect(url_for("dashboard"))
    return render_template(
        "capture.html", student=student,
        total_images=config.TOTAL_CAPTURE_IMAGES,
        interval_ms=config.CAPTURE_INTERVAL_MS,
        poses=[{"instruction": p[1], "target": p[2]} for p in config.CAPTURE_POSES])


@app.route("/api/capture_frame", methods=["POST"])
@student_required
def api_capture_frame():
    student_id = session["student_id"]
    student = database.find_student(student_id=student_id)
    if student["registration_status"] != "registered":
        return jsonify({"done": True, "accepted": False,
                        "reason": "Dataset already collected."})

    if not _liveness_valid("capture"):
        return jsonify({"needs_liveness": True, "accepted": False,
                        "done": False,
                        "reason": "Liveness check required"})

    frame = _decode_frame(request.json.get("image", ""))

    with _capture_lock:
        state = _capture_sessions.setdefault(
            student_id, {"count": 0, "pose_index": 0, "pose_count": 0,
                         "fingerprint": None})
        pose_index, pose_count = state["pose_index"], state["pose_count"]
        prev_fp = state["fingerprint"]

    pose_key, _, pose_target, require_eyes = config.CAPTURE_POSES[pose_index]

    ok, reason, crop_rgb, fingerprint = detector.check_frame(
        frame, prev_fingerprint=prev_fp, pose=pose_key,
        require_eyes=require_eyes)

    accepted = False
    if ok:
        folder = os.path.join(config.DATASET_DIR, student_id)
        os.makedirs(folder, exist_ok=True)
        with _capture_lock:
            state["count"] += 1
            state["pose_count"] += 1
            state["fingerprint"] = fingerprint
            count = state["count"]
            if state["pose_count"] >= pose_target and \
                    state["pose_index"] < len(config.CAPTURE_POSES) - 1:
                state["pose_index"] += 1
                state["pose_count"] = 0
            pose_index, pose_count = state["pose_index"], state["pose_count"]
        cv2.imwrite(os.path.join(folder, f"{count:03d}.jpg"),
                    cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        accepted = True
        reason = "Captured"
    else:
        with _capture_lock:
            count = state["count"]

    done = count >= config.TOTAL_CAPTURE_IMAGES
    if done:
        database.update_student_status(student_id, "dataset_collected",
                                       num_images=count)
        with _capture_lock:
            _capture_sessions.pop(student_id, None)
        _liveness_consume("capture")

    return jsonify({
        "accepted": accepted,
        "reason": reason,
        "count": count,
        "total": config.TOTAL_CAPTURE_IMAGES,
        "pose": _pose_info(pose_index, pose_count),
        "done": done,
    })


@app.route("/capture/complete")
@student_required
def capture_complete():
    student = database.find_student(student_id=session["student_id"])
    return render_template("capture_complete.html", student=student)


# ---------------------------------------------------------------------------
# Student dashboard + attendance
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@student_required
def dashboard():
    student = database.find_student(student_id=session["student_id"])
    records = database.attendance_records(student_id=student["student_id"])
    return render_template(
        "dashboard.html", student=student, records=records,
        subjects=database.all_subjects(),
        model_ready=recognize.model_exists(),
        enrolled=student["student_id"] in database.load_all_embeddings())


@app.route("/attendance")
@student_required
def attendance():
    student = database.find_student(student_id=session["student_id"])
    subjects = database.all_subjects()
    if not subjects:
        return render_template("message.html", title="No Subjects",
                               message="No subjects are configured. Ask the "
                                       "administrator to add subjects first.")
    subject = request.args.get("subject", subjects[0])
    if subject not in subjects:
        subject = subjects[0]

    if not recognize.model_exists():
        return render_template("message.html", title="Model Not Trained",
                               message="The FaceNet model has not been trained "
                                       "yet. Please ask the administrator to "
                                       "train the model first.")
    if student["student_id"] not in database.load_all_embeddings():
        return render_template("message.html", title="Not Enrolled",
                               message="Your face embedding is not in the "
                                       "database yet. The administrator must "
                                       "(re)train the model after your "
                                       "dataset was collected.")

    today = datetime.now().strftime("%Y-%m-%d")
    if database.attendance_exists(student["student_id"], subject, today):
        return render_template("message.html", title="Attendance Already Marked",
                               message=f"Attendance for {subject} on {today} "
                                       "already exists. No duplicate record "
                                       "was created.")
    return render_template("attendance.html", student=student, subject=subject)


@app.route("/api/recognize", methods=["POST"])
@student_required
def api_recognize():
    student = database.find_student(student_id=session["student_id"])
    subject = request.json.get("subject", "")
    if subject not in database.all_subjects():
        return jsonify({"status": "error", "message": "Unknown subject."})
    if not recognize.model_exists():
        return jsonify({"status": "error", "message": "Model not trained."})
    if not _liveness_valid("attendance"):
        return jsonify({"status": "needs_liveness",
                        "message": "Liveness check required."})

    frame = _decode_frame(request.json.get("image", ""))
    if frame is None:
        return jsonify({"status": "retry", "message": "Bad frame."})

    face, info = detector.extract_face_for_recognition(frame)
    if face is None:
        return jsonify({"status": "retry", "message": info})

    # anti-spoofing on the exact frame that would mark attendance
    from face_utils import antispoof
    if antispoof.available():
        is_real, real_prob = antispoof.check(frame, info)
        if not is_real:
            return jsonify({"status": "spoof",
                            "real_prob": round(real_prob, 3),
                            "message": "Photo or screen detected — "
                                       "show your real face"})

    embedding = recognize.embed_face(face)
    matched_id, similarity = recognize.match_embedding(embedding)
    similarity = round(similarity, 3)

    if matched_id is None:
        return jsonify({"status": "unknown", "similarity": similarity,
                        "message": "Unknown Person — Attendance Not Marked"})

    if config.REQUIRE_LOGIN_MATCH and matched_id != student["student_id"]:
        other = database.find_student(student_id=matched_id)
        name = other["full_name"] if other else matched_id
        return jsonify({"status": "mismatch", "similarity": similarity,
                        "message": f"Face recognized as {name}, which does not "
                                   "match the logged-in account. Attendance "
                                   "not marked."})

    matched = database.find_student(student_id=matched_id)
    created = database.mark_attendance(
        matched_id, matched["full_name"], subject, similarity)
    if not created:
        return jsonify({"status": "duplicate", "similarity": similarity,
                        "message": "Attendance Already Marked"})

    _liveness_consume("attendance")   # one liveness pass = one marking
    return jsonify({"status": "marked", "similarity": similarity,
                    "name": matched["full_name"],
                    "message": f"Welcome {matched['full_name']} — "
                               "Attendance Marked Successfully"})


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html", error=None)
    if (request.form.get("username") == config.ADMIN_USERNAME
            and request.form.get("password") == config.ADMIN_PASSWORD):
        session["is_admin"] = True
        return redirect(url_for("admin_dashboard"))
    return render_template("admin_login.html", error="Invalid credentials.")


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    students = database.all_students()
    embeddings = database.load_all_embeddings()
    for s in students:
        s["enrolled"] = s["student_id"] in embeddings
        folder = os.path.join(config.DATASET_DIR, s["student_id"])
        s["images_on_disk"] = (
            len([f for f in os.listdir(folder) if f.endswith(".jpg")])
            if os.path.isdir(folder) else 0)
    records = database.attendance_records()
    return render_template(
        "admin_dashboard.html", students=students, records=records,
        subjects=database.all_subjects(),
        semesters=database.all_semesters(),
        model_ready=recognize.model_exists(),
        history_plot=os.path.exists(config.HISTORY_PLOT_PATH))


@app.route("/api/admin/add_subject", methods=["POST"])
@admin_required
def api_admin_add_subject():
    ok, message = database.add_subject(request.json.get("name", ""))
    return jsonify({"ok": ok, "message": message,
                    "subjects": database.all_subjects()})


@app.route("/api/admin/delete_subject", methods=["POST"])
@admin_required
def api_admin_delete_subject():
    ok, message = database.delete_subject(request.json.get("name", ""))
    return jsonify({"ok": ok, "message": message,
                    "subjects": database.all_subjects()})


@app.route("/api/admin/add_semester", methods=["POST"])
@admin_required
def api_admin_add_semester():
    ok, message = database.add_semester(request.json.get("name", ""))
    return jsonify({"ok": ok, "message": message,
                    "semesters": database.all_semesters()})


@app.route("/api/admin/delete_semester", methods=["POST"])
@admin_required
def api_admin_delete_semester():
    ok, message = database.delete_semester(request.json.get("name", ""))
    return jsonify({"ok": ok, "message": message,
                    "semesters": database.all_semesters()})


@app.route("/api/admin/set_student_semester", methods=["POST"])
@admin_required
def api_admin_set_student_semester():
    """Promote/correct a student's semester (students never re-register,
    so the admin updates this each term)."""
    student_id = request.json.get("student_id", "")
    semester = request.json.get("semester", "")
    if semester not in database.all_semesters():
        return jsonify({"ok": False, "message": "Unknown semester."})
    if not database.set_student_semester(student_id, semester):
        return jsonify({"ok": False, "message": "Student not found."})
    return jsonify({"ok": True,
                    "message": f"Student {student_id} moved to semester {semester}."})


@app.route("/api/admin/train", methods=["POST"])
@admin_required
def api_admin_train():
    ok, message = trainer.start_training()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/train_status")
@admin_required
def api_train_status():
    return jsonify(trainer.get_state())


@app.route("/api/admin/reset_student", methods=["POST"])
@admin_required
def api_admin_reset_student():
    """Reset a student's enrollment: delete dataset + embedding so they can
    capture a fresh dataset. Account and attendance history are kept."""
    student_id = request.json.get("student_id", "")
    student = database.find_student(student_id=student_id)
    if not student:
        return jsonify({"ok": False, "message": "Student not found."})
    folder = os.path.join(config.DATASET_DIR, student_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    database.delete_embedding(student_id)
    database.update_student_status(student_id, "registered", num_images=0)
    return jsonify({"ok": True,
                    "message": f"Enrollment reset for {student['full_name']}. "
                               "They must capture a new dataset."})


@app.route("/admin/history_plot")
@admin_required
def admin_history_plot():
    if not os.path.exists(config.HISTORY_PLOT_PATH):
        return "No training history yet.", 404
    return send_file(config.HISTORY_PLOT_PATH, mimetype="image/png")


@app.route("/admin/export.csv")
@admin_required
def admin_export_csv():
    records = database.attendance_records(limit=100000)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Student ID", "Name", "Date", "Time", "Subject",
                     "Confidence", "Status"])
    for r in records:
        writer.writerow([r["student_id"], r["name"], r["date"], r["time"],
                         r["subject"], r["confidence"], r["status"]])
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype="text/csv", as_attachment=True,
                     download_name="attendance_export.csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
