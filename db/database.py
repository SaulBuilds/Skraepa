import os
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                host=os.environ.get('PGHOST'),
                database=os.environ.get('PGDATABASE'),
                user=os.environ.get('PGUSER'),
                password=os.environ.get('PGPASSWORD'),
                port=os.environ.get('PGPORT')
            )
            logger.info("Successfully connected to the database")
            self.create_tables()
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def create_tables(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS harvested_data (
                        id SERIAL PRIMARY KEY,
                        url TEXT NOT NULL,
                        content TEXT,
                        raw_content TEXT,
                        analysis JSONB,
                        processing_metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.conn.commit()
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise

    def save_data(self, url: str, content: str, raw_content: str, analysis: dict, processing_metadata: dict | None = None) -> int:
        """Save data with raw content and processing metadata"""
        try:
            if processing_metadata is None:
                processing_metadata = {
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "processing_status": "completed",
                    "processing_steps": []
                }

            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO harvested_data 
                    (url, content, raw_content, analysis, processing_metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (url, content, raw_content, Json(analysis), Json(processing_metadata))
                )
                self.conn.commit()
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise

    def get_all_data(self, include_raw: bool = False) -> list:
        """Get all data with option to include raw content"""
        try:
            with self.conn.cursor() as cur:
                if include_raw:
                    cur.execute("""
                        SELECT id, url, content, raw_content, analysis, processing_metadata, created_at 
                        FROM harvested_data 
                        ORDER BY created_at DESC
                    """)
                else:
                    cur.execute("""
                        SELECT id, url, content, analysis, processing_metadata, created_at 
                        FROM harvested_data 
                        ORDER BY created_at DESC
                    """)
                return cur.fetchall() or []
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            raise

    def get_data_by_date_range(self, start_date: str, end_date: str, include_raw: bool = False) -> list:
        """Get data within a date range"""
        try:
            with self.conn.cursor() as cur:
                if include_raw:
                    cur.execute("""
                        SELECT id, url, content, raw_content, analysis, processing_metadata, created_at 
                        FROM harvested_data 
                        WHERE created_at BETWEEN %s AND %s 
                        ORDER BY created_at DESC
                    """, (start_date, end_date))
                else:
                    cur.execute("""
                        SELECT id, url, content, analysis, processing_metadata, created_at 
                        FROM harvested_data 
                        WHERE created_at BETWEEN %s AND %s 
                        ORDER BY created_at DESC
                    """, (start_date, end_date))
                return cur.fetchall() or []
        except Exception as e:
            logger.error(f"Failed to fetch data by date range: {str(e)}")
            raise

    def update_processing_metadata(self, data_id: int, metadata: dict) -> None:
        """Update processing metadata for a specific entry"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE harvested_data 
                    SET processing_metadata = %s 
                    WHERE id = %s
                    """,
                    (Json(metadata), data_id)
                )
                self.conn.commit()
                logger.info(f"Updated processing metadata for data_id: {data_id}")
        except Exception as e:
            logger.error(f"Failed to update processing metadata: {str(e)}")
            raise

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Database connection closed")
