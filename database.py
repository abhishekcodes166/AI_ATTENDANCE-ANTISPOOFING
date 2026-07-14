"""SQLite database layer: students, embeddings, attendance."""

import sqlite3
from datetime import datetime

import numpy as np

import config


def get_db():
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                roll_number TEXT NOT NULL,
                department TEXT NOT NULL,
                semester TEXT NOT NULL,
                section TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                registration_status TEXT NOT NULL DEFAULT 'registered',
                num_images INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                student_id TEXT PRIMARY KEY
                    REFERENCES students(student_id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL COLLATE NOCASE
            );

            CREATE TABLE IF NOT EXISTS semesters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL COLLATE NOCASE
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL
                    REFERENCES students(student_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                subject TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                UNIQUE (student_id, subject, date)
            );
            """
        )


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def create_student(student_id, full_name, roll_number, department, semester,
                   section, email, password_hash):
    with get_db() as db:
        db.execute(
            """INSERT INTO students
               (student_id, full_name, roll_number, department, semester,
                section, email, password_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (student_id, full_name, roll_number, department, semester,
             section, email, password_hash, datetime.now().isoformat()),
        )


def find_student(student_id=None, email=None):
    with get_db() as db:
        if student_id is not None and email is not None:
            row = db.execute(
                "SELECT * FROM students WHERE student_id = ? OR email = ?",
                (student_id, email)).fetchone()
        elif student_id is not None:
            row = db.execute(
                "SELECT * FROM students WHERE student_id = ?",
                (student_id,)).fetchone()
        else:
            row = db.execute(
                "SELECT * FROM students WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None


def all_students():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM students ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]


def update_student_status(student_id, status, num_images=None):
    with get_db() as db:
        if num_images is None:
            db.execute(
                "UPDATE students SET registration_status = ? WHERE student_id = ?",
                (status, student_id))
        else:
            db.execute(
                """UPDATE students SET registration_status = ?, num_images = ?
                   WHERE student_id = ?""",
                (status, num_images, student_id))


def delete_student(student_id):
    with get_db() as db:
        db.execute("DELETE FROM students WHERE student_id = ?", (student_id,))


# ---------------------------------------------------------------------------
# Embeddings (128-D float32 vectors stored as raw bytes)
# ---------------------------------------------------------------------------

def save_embedding(student_id, vector):
    vector = np.asarray(vector, dtype=np.float32)
    with get_db() as db:
        db.execute(
            """INSERT INTO embeddings (student_id, embedding, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(student_id) DO UPDATE SET
                   embedding = excluded.embedding,
                   updated_at = excluded.updated_at""",
            (student_id, vector.tobytes(), datetime.now().isoformat()),
        )


def load_all_embeddings():
    """Return {student_id: 128-D numpy vector} for every enrolled student."""
    with get_db() as db:
        rows = db.execute("SELECT student_id, embedding FROM embeddings").fetchall()
    return {
        r["student_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
        for r in rows
    }


def delete_embedding(student_id):
    with get_db() as db:
        db.execute("DELETE FROM embeddings WHERE student_id = ?", (student_id,))


# ---------------------------------------------------------------------------
# Subjects (managed by the admin; defaults seeded on first run)
# ---------------------------------------------------------------------------

def seed_default_subjects():
    with get_db() as db:
        count = db.execute("SELECT COUNT(*) AS n FROM subjects").fetchone()["n"]
        if count == 0:
            db.executemany("INSERT INTO subjects (name) VALUES (?)",
                           [(s,) for s in config.DEFAULT_SUBJECTS])


def all_subjects():
    with get_db() as db:
        rows = db.execute("SELECT name FROM subjects ORDER BY name").fetchall()
        return [r["name"] for r in rows]


def add_subject(name):
    name = " ".join(name.split()).strip()
    if not name:
        return False, "Subject name cannot be empty."
    if len(name) > 80:
        return False, "Subject name is too long."
    try:
        with get_db() as db:
            db.execute("INSERT INTO subjects (name) VALUES (?)", (name,))
        return True, f"Subject '{name}' added."
    except sqlite3.IntegrityError:
        return False, f"Subject '{name}' already exists."


def delete_subject(name):
    with get_db() as db:
        cur = db.execute("DELETE FROM subjects WHERE name = ?", (name,))
        if cur.rowcount == 0:
            return False, "Subject not found."
    return True, f"Subject '{name}' removed."


# ---------------------------------------------------------------------------
# Semesters (managed by the admin; defaults 1-8 seeded on first run)
# ---------------------------------------------------------------------------

def seed_default_semesters():
    with get_db() as db:
        count = db.execute("SELECT COUNT(*) AS n FROM semesters").fetchone()["n"]
        if count == 0:
            db.executemany("INSERT INTO semesters (name) VALUES (?)",
                           [(s,) for s in config.DEFAULT_SEMESTERS])


def all_semesters():
    with get_db() as db:
        rows = db.execute("SELECT name FROM semesters").fetchall()
    # numeric-aware ordering: 1, 2, ... 10 before any non-numeric names
    return sorted((r["name"] for r in rows),
                  key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))


def add_semester(name):
    name = " ".join(str(name).split()).strip()
    if not name:
        return False, "Semester cannot be empty."
    if len(name) > 20:
        return False, "Semester name is too long."
    try:
        with get_db() as db:
            db.execute("INSERT INTO semesters (name) VALUES (?)", (name,))
        return True, f"Semester '{name}' added."
    except sqlite3.IntegrityError:
        return False, f"Semester '{name}' already exists."


def delete_semester(name):
    with get_db() as db:
        cur = db.execute("DELETE FROM semesters WHERE name = ?", (name,))
        if cur.rowcount == 0:
            return False, "Semester not found."
    return True, f"Semester '{name}' removed."


def set_student_semester(student_id, semester):
    with get_db() as db:
        cur = db.execute("UPDATE students SET semester = ? WHERE student_id = ?",
                         (semester, student_id))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------

def attendance_exists(student_id, subject, date):
    with get_db() as db:
        row = db.execute(
            """SELECT 1 FROM attendance
               WHERE student_id = ? AND subject = ? AND date = ?""",
            (student_id, subject, date)).fetchone()
        return row is not None


def mark_attendance(student_id, name, subject, confidence):
    now = datetime.now()
    date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    if attendance_exists(student_id, subject, date):
        return False
    with get_db() as db:
        db.execute(
            """INSERT INTO attendance
               (student_id, name, date, time, subject, confidence, status)
               VALUES (?, ?, ?, ?, ?, ?, 'Present')""",
            (student_id, name, date, time, subject, confidence),
        )
    return True


def attendance_records(student_id=None, limit=500):
    with get_db() as db:
        if student_id:
            rows = db.execute(
                """SELECT * FROM attendance WHERE student_id = ?
                   ORDER BY date DESC, time DESC LIMIT ?""",
                (student_id, limit)).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM attendance ORDER BY date DESC, time DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]


init_db()
seed_default_subjects()
seed_default_semesters()
