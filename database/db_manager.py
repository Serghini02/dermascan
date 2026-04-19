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
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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

        # Migración: añadir image_data si no existe (bases de datos antiguas)
        try:
            c.execute("ALTER TABLE consultations ADD COLUMN image_data TEXT")
        except Exception:
            pass  # La columna ya existe

        conn.commit()
        conn.close()

    def add_consultation(self, image_path=None, image_data=None, cnn_diagnosis=None,
                         cnn_confidence=0.0, cnn_probabilities=None,
                         symptoms=None, abcde_scores=None,
                         drl_diagnosis=None, risk_level=None,
                         questions_asked=0, final_recommendation=None):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO consultations
            (timestamp, image_path, image_data, cnn_diagnosis, cnn_confidence,
             cnn_probabilities, symptoms, abcde_scores,
             drl_diagnosis, risk_level, questions_asked, final_recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(), image_path, image_data, cnn_diagnosis,
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

    def delete_consultation(self, consultation_id):
        """Elimina una consulta del historial por su ID."""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM consultations WHERE id = ?", (consultation_id,))
        affected = c.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def get_consultations(self, limit=50):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM consultations ORDER BY id DESC LIMIT ?", (limit,))
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
