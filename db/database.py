import os
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import Json

class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.environ.get('PGHOST'),
            database=os.environ.get('PGDATABASE'),
            user=os.environ.get('PGUSER'),
            password=os.environ.get('PGPASSWORD'),
            port=os.environ.get('PGPORT')
        )
        self.create_tables()

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS harvested_data (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    content TEXT,
                    analysis JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

    def save_data(self, url: str, content: str, analysis: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvested_data (url, content, analysis)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (url, content, Json(analysis))
            )
            self.conn.commit()
            return cur.fetchone()[0]

    def get_all_data(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM harvested_data ORDER BY created_at DESC")
            return cur.fetchall()

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
