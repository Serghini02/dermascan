"""
Base de datos SQLite para DermaScan.
Almacena consultas, diagnósticos e historial de pacientes.
"""
import sqlite3
import json
import os
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path="database/dermascan.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                image_data TEXT,
                cnn_diagnosis TEXT,
                cnn_confidence REAL,
                cnn_probabilities TEXT,
                symptoms TEXT,
                abcde_scores TEXT,
                drl_diagnosis TEXT,
                risk_level TEXT,
                questions_asked INTEGER DEFAULT 0,
                final_recommendation TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS ham10000_metadata (
                image_id TEXT PRIMARY KEY,
                lesion_id TEXT,
                dx TEXT,
                dx_type TEXT,
                age REAL,
                sex TEXT,
                localization TEXT
            )
        """)

        # Migraciones para bases de datos antiguas (pre-EpidermAI)
        try:
            c.execute("ALTER TABLE consultations ADD COLUMN image_data TEXT")
        except Exception:
            pass  # La columna ya existe

        c.execute("PRAGMA table_info(consultations)")
        columns = [row[1] for row in c.fetchall()]
        if "user_id" not in columns:
            # Los registros previos no tienen propietario: se descartan al
            # introducir el aislamiento de datos por usuario.
            c.execute("DELETE FROM consultations")
            c.execute("ALTER TABLE consultations ADD COLUMN user_id INTEGER")

        conn.commit()
        conn.close()

    # =========================================================================
    # USUARIOS Y TOKENS
    # =========================================================================
    def create_user(self, name, email, password_hash):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email.lower(), password_hash, datetime.now().isoformat()),
        )
        conn.commit()
        uid = c.lastrowid
        conn.close()
        return uid

    def get_user_by_email(self, email):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_user_by_id(self, user_id):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def save_refresh_token(self, user_id, token_hash, expires_at):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (user_id, token_hash, expires_at, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def get_refresh_token(self, token_hash):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM refresh_tokens WHERE token_hash = ?", (token_hash,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def revoke_refresh_token(self, token_hash):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ?", (token_hash,))
        conn.commit()
        conn.close()

    # =========================================================================
    # CONSULTAS (aisladas por usuario)
    # =========================================================================
    def add_consultation(self, user_id, image_path=None, image_data=None, cnn_diagnosis=None,
                         cnn_confidence=0.0, cnn_probabilities=None,
                         symptoms=None, abcde_scores=None,
                         drl_diagnosis=None, risk_level=None,
                         questions_asked=0, final_recommendation=None):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO consultations
            (user_id, timestamp, image_path, image_data, cnn_diagnosis, cnn_confidence,
             cnn_probabilities, symptoms, abcde_scores,
             drl_diagnosis, risk_level, questions_asked, final_recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, datetime.now().isoformat(), image_path, image_data, cnn_diagnosis,
            cnn_confidence,
            json.dumps(cnn_probabilities) if cnn_probabilities else None,
            json.dumps(symptoms) if symptoms else None,
            json.dumps(abcde_scores) if abcde_scores else None,
            drl_diagnosis, risk_level, questions_asked, final_recommendation
        ))
        conn.commit()
        cid = c.lastrowid
        conn.close()
        return cid

    def delete_consultation(self, user_id, consultation_id):
        """Elimina una consulta del historial, solo si pertenece al usuario."""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM consultations WHERE id = ? AND user_id = ?", (consultation_id, user_id))
        affected = c.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def get_consultations(self, user_id, limit=50):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute(
            "SELECT * FROM consultations WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = []
        for r in c.fetchall():
            d = dict(r)
            for key in ("cnn_probabilities", "symptoms", "abcde_scores"):
                if d.get(key):
                    d[key] = json.loads(d[key])
            rows.append(d)
        conn.close()
        return rows

    def load_ham10000_metadata(self, csv_path):
        """Carga el CSV de metadata del HAM10000 en la BD."""
        import csv
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM ham10000_metadata")
        if c.fetchone()[0] > 0:
            print("[DB] Metadata HAM10000 ya cargada.")
            conn.close()
            return

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                rows.append((
                    row.get("image_id", ""),
                    row.get("lesion_id", ""),
                    row.get("dx", ""),
                    row.get("dx_type", ""),
                    float(row["age"]) if row.get("age") else None,
                    row.get("sex", ""),
                    row.get("localization", ""),
                ))
            c.executemany("""
                INSERT OR IGNORE INTO ham10000_metadata
                (image_id, lesion_id, dx, dx_type, age, sex, localization)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)

        conn.commit()
        print(f"[DB] Cargadas {len(rows)} entradas de metadata HAM10000.")
        conn.close()

    def get_ham10000_data(self):
        """Obtiene todos los registros HAM10000 para entrenar el DRL."""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM ham10000_metadata")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_ham10000_stats(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT dx, COUNT(*) as count FROM ham10000_metadata GROUP BY dx ORDER BY count DESC")
        stats = {r["dx"]: r["count"] for r in c.fetchall()}
        c.execute("SELECT COUNT(*) as total FROM ham10000_metadata")
        total = c.fetchone()["total"]
        conn.close()
        return {"total": total, "by_diagnosis": stats}

    def count_consultations(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM consultations")
        n = c.fetchone()[0]
        conn.close()
        return n
