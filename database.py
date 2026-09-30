"""
OptiScholar — Database & Authentication Module
================================================
SQLite-based persistent storage for:
  - User accounts (signup/login)
  - Student profiles
  - Scholarship interactions (applied/saved/ignored)
  - Saved scholarships
  - Deadline reminders

Usage:
    from database import Database
    db = Database()
    db.create_user("nada@bue.edu", "password123", "Nada Mohamed")
    user = db.login("nada@bue.edu", "password123")
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, date
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optischolar.db")


def _hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with salt using SHA-256."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return hashed, salt


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create all tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                full_name     TEXT NOT NULL,
                created_at    TEXT DEFAULT (datetime('now')),
                last_login    TEXT
            );

            -- Student profiles
            CREATE TABLE IF NOT EXISTS profiles (
                user_id           INTEGER PRIMARY KEY,
                final_gpa         REAL,
                degree_level      TEXT,
                field_of_study    TEXT,
                institution       TEXT,
                age               INTEGER,
                gender            TEXT,
                ses_category      TEXT,
                household_income  REAL,
                financial_need    INTEGER DEFAULT 0,
                is_international  INTEGER DEFAULT 0,
                updated_at        TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- Scholarship interactions (applied/saved/ignored)
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL,
                scholarship_id    TEXT NOT NULL,
                scholarship_title TEXT,
                action            TEXT CHECK(action IN ('applied','saved','ignored')),
                match_score       REAL,
                amount            REAL,
                deadline          TEXT,
                link              TEXT,
                scholarship_type  TEXT,
                created_at        TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, scholarship_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- Cached recommendations
            CREATE TABLE IF NOT EXISTS recommendations (
                rec_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL,
                scholarship_id    TEXT NOT NULL,
                scholarship_title TEXT,
                match_score       REAL,
                scholarship_type  TEXT,
                amount            REAL,
                deadline          TEXT,
                link              TEXT,
                description       TEXT,
                shap_values       TEXT,  -- JSON string
                generated_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- Deadline reminders
            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL,
                scholarship_id    TEXT NOT NULL,
                scholarship_title TEXT,
                deadline_date     TEXT,
                amount            REAL,
                link              TEXT,
                is_dismissed      INTEGER DEFAULT 0,
                created_at        TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
            """)

    # ── AUTH ──────────────────────────────────────────────────

    def create_user(self, email: str, password: str,
                     full_name: str) -> tuple:
        """
        Create new user account.
        Returns (success: bool, message: str, user_id: int|None)
        """
        if not email or not password or not full_name:
            return False, "All fields are required.", None
        if len(password) < 6:
            return False, "Password must be at least 6 characters.", None
        if "@" not in email:
            return False, "Invalid email address.", None

        hashed, salt = _hash_password(password)
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (email, password_hash, salt, full_name) "
                    "VALUES (?, ?, ?, ?)",
                    (email.lower().strip(), hashed, salt, full_name.strip())
                )
                user_id = cursor.lastrowid
                # Create empty profile
                conn.execute(
                    "INSERT INTO profiles (user_id) VALUES (?)",
                    (user_id,)
                )
            return True, "Account created successfully!", user_id
        except sqlite3.IntegrityError:
            return False, "Email already registered. Please log in.", None

    def login(self, email: str, password: str) -> tuple:
        """
        Authenticate user.
        Returns (success: bool, message: str, user_dict|None)
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email.lower().strip(),)
            ).fetchone()

        if not row:
            return False, "Email not found.", None

        hashed, _ = _hash_password(password, row["salt"])
        if hashed != row["password_hash"]:
            return False, "Incorrect password.", None

        # Update last login
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET last_login = datetime('now') WHERE user_id = ?",
                (row["user_id"],)
            )

        return True, "Welcome back!", dict(row)

    def get_user(self, user_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── PROFILE ───────────────────────────────────────────────

    def save_profile(self, user_id: int, profile: dict) -> bool:
        """Save or update student profile."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO profiles
                    (user_id, final_gpa, degree_level, field_of_study,
                     institution, age, gender, ses_category,
                     household_income, financial_need, is_international,
                     updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(user_id) DO UPDATE SET
                    final_gpa        = excluded.final_gpa,
                    degree_level     = excluded.degree_level,
                    field_of_study   = excluded.field_of_study,
                    institution      = excluded.institution,
                    age              = excluded.age,
                    gender           = excluded.gender,
                    ses_category     = excluded.ses_category,
                    household_income = excluded.household_income,
                    financial_need   = excluded.financial_need,
                    is_international = excluded.is_international,
                    updated_at       = datetime('now')
            """, (
                user_id,
                profile.get("final_gpa"),
                profile.get("degree_level"),
                profile.get("_field_of_study",""),
                profile.get("_institution",""),
                profile.get("age"),
                profile.get("gender"),
                profile.get("ses_category","Middle"),
                profile.get("household_income"),
                int(profile.get("financial_need",0)),
                int(profile.get("International",0)),
            ))
        return True

    def get_profile(self, user_id: int) -> Optional[dict]:
        """Load student profile."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if not row:
            return None
        p = dict(row)
        # Map to app schema
        p["gpa_proxy"]    = p.get("final_gpa")
        p["International"]= p.get("is_international", 0)
        return p

    # ── INTERACTIONS (Applied/Saved/Ignored) ──────────────────

    def save_interaction(self, user_id: int, scholarship_id: str,
                          action: str, scholarship_data: dict) -> bool:
        """
        Save student interaction with a scholarship.
        action: 'applied' | 'saved' | 'ignored'
        This feeds back into model retraining.
        """
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO interactions
                    (user_id, scholarship_id, scholarship_title, action,
                     match_score, amount, deadline, link, scholarship_type)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(user_id, scholarship_id) DO UPDATE SET
                    action     = excluded.action,
                    created_at = datetime('now')
            """, (
                user_id,
                scholarship_id,
                scholarship_data.get("scholarship_title",""),
                action,
                scholarship_data.get("transfer_score", 0),
                scholarship_data.get("funding_amount_raw", 0),
                scholarship_data.get("deadline_date",""),
                scholarship_data.get("link",""),
                scholarship_data.get("scholarship_type",""),
            ))
        return True

    def get_interactions(self, user_id: int,
                          action: str = None) -> list:
        """Get all interactions for a user, optionally filtered by action."""
        query = "SELECT * FROM interactions WHERE user_id = ?"
        params = [user_id]
        if action:
            query += " AND action = ?"
            params.append(action)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_interaction_action(self, user_id: int,
                                scholarship_id: str) -> Optional[str]:
        """Get current action for a specific scholarship."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT action FROM interactions "
                "WHERE user_id = ? AND scholarship_id = ?",
                (user_id, scholarship_id)
            ).fetchone()
        return row["action"] if row else None

    def export_interactions_for_training(self) -> list:
        """
        Export all interactions as implicit ratings for model retraining.
        Maps: applied=1.0, saved=0.7, ignored=0.1
        """
        rating_map = {"applied": 1.0, "saved": 0.7, "ignored": 0.1}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_id, scholarship_id, action FROM interactions"
            ).fetchall()
        return [
            {
                "student_id":     f"USER_{r['user_id']}",
                "scholarship_id": r["scholarship_id"],
                "rating":         rating_map.get(r["action"], 0.5),
                "source":         "user_feedback",
            }
            for r in rows
        ]

    # ── RECOMMENDATIONS CACHE ─────────────────────────────────

    def save_recommendations(self, user_id: int,
                              recs: list) -> bool:
        """Cache recommendations for a user."""
        import json
        with self._connect() as conn:
            # Clear old recommendations
            conn.execute(
                "DELETE FROM recommendations WHERE user_id = ?", (user_id,)
            )
            for rec in recs:
                shap = rec.get("_shap_values")
                shap_json = json.dumps(shap.tolist()) \
                            if shap is not None else None
                conn.execute("""
                    INSERT INTO recommendations
                        (user_id, scholarship_id, scholarship_title,
                         match_score, scholarship_type, amount,
                         deadline, link, description, shap_values)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    user_id,
                    rec.get("scholarship_id",""),
                    rec.get("scholarship_title",""),
                    rec.get("transfer_score", 0),
                    rec.get("scholarship_type",""),
                    rec.get("funding_amount_raw", 0),
                    rec.get("deadline_date",""),
                    rec.get("link",""),
                    rec.get("description_cleaned","")[:300],
                    shap_json,
                ))
        return True

    def get_cached_recommendations(self, user_id: int) -> list:
        """Load cached recommendations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE user_id = ? "
                "ORDER BY match_score DESC",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── REMINDERS ─────────────────────────────────────────────

    def add_reminder(self, user_id: int,
                      scholarship_data: dict) -> bool:
        """Add deadline reminder for a scholarship."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO reminders
                    (user_id, scholarship_id, scholarship_title,
                     deadline_date, amount, link)
                VALUES (?,?,?,?,?,?)
            """, (
                user_id,
                scholarship_data.get("scholarship_id",""),
                scholarship_data.get("scholarship_title",""),
                scholarship_data.get("deadline_date",""),
                scholarship_data.get("funding_amount_raw", 0),
                scholarship_data.get("link",""),
            ))
        return True

    def get_upcoming_reminders(self, user_id: int,
                                days_ahead: int = 30) -> list:
        """Get non-dismissed reminders with deadline within days_ahead."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM reminders
                WHERE user_id = ?
                  AND is_dismissed = 0
                  AND deadline_date != ''
                  AND deadline_date >= date('now')
                  AND deadline_date <= date('now', ? || ' days')
                ORDER BY deadline_date ASC
            """, (user_id, str(days_ahead))).fetchall()
        return [dict(r) for r in rows]

    def dismiss_reminder(self, reminder_id: int) -> bool:
        with self._connect() as conn:
            conn.execute(
                "UPDATE reminders SET is_dismissed = 1 "
                "WHERE reminder_id = ?",
                (reminder_id,)
            )
        return True

    # ── DASHBOARD STATS ───────────────────────────────────────

    def get_dashboard_stats(self, user_id: int) -> dict:
        """Get summary stats for user dashboard."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as n FROM recommendations WHERE user_id=?",
                (user_id,)
            ).fetchone()["n"]

            applied = conn.execute(
                "SELECT COUNT(*) as n FROM interactions "
                "WHERE user_id=? AND action='applied'",
                (user_id,)
            ).fetchone()["n"]

            saved = conn.execute(
                "SELECT COUNT(*) as n FROM interactions "
                "WHERE user_id=? AND action='saved'",
                (user_id,)
            ).fetchone()["n"]

            reminders = conn.execute(
                "SELECT COUNT(*) as n FROM reminders "
                "WHERE user_id=? AND is_dismissed=0 "
                "AND deadline_date >= date('now')",
                (user_id,)
            ).fetchone()["n"]

        return {
            "total_recommendations": total,
            "applied":               applied,
            "saved":                 saved,
            "reminders":             reminders,
        }


if __name__ == "__main__":
    # Quick test
    db = Database()
    ok, msg, uid = db.create_user("test@test.com","password123","Test User")
    print(f"Create: {ok} — {msg} — ID:{uid}")
    ok, msg, user = db.login("test@test.com","password123")
    print(f"Login: {ok} — {msg} — {user['full_name']}")
    ok, msg, _ = db.login("test@test.com","wrongpass")
    print(f"Wrong pass: {ok} — {msg}")
    print("Database OK ✅")
