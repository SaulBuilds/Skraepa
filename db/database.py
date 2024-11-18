import os
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import logging
import uuid

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
        """Create database tables with proper error handling and order"""
        try:
            with self.conn.cursor() as cur:
                # Create base harvested_data table
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

                # Add session management columns to harvested_data
                cur.execute("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='harvested_data' AND column_name='session_id'
                        ) THEN
                            ALTER TABLE harvested_data 
                            ADD COLUMN session_id VARCHAR(64),
                            ADD COLUMN is_temporary BOOLEAN DEFAULT TRUE,
                            ADD COLUMN last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                        END IF;
                    END $$;
                """)

                # Create base harvested_media table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS harvested_media (
                        id SERIAL PRIMARY KEY,
                        harvested_data_id INTEGER REFERENCES harvested_data(id) ON DELETE CASCADE,
                        media_type VARCHAR(10) NOT NULL,
                        url TEXT NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Add session management columns to harvested_media
                cur.execute("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='harvested_media' AND column_name='session_id'
                        ) THEN
                            ALTER TABLE harvested_media 
                            ADD COLUMN session_id VARCHAR(64),
                            ADD COLUMN is_temporary BOOLEAN DEFAULT TRUE;
                        END IF;
                    END $$;
                """)

                # Create indexes for better performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_temp ON harvested_data(session_id, is_temporary);
                    CREATE INDEX IF NOT EXISTS idx_media_session_temp ON harvested_media(session_id, is_temporary);
                """)
                
                self.conn.commit()
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            self.conn.rollback()
            raise

    def save_data(self, url: str, content: str, raw_content: str, analysis: dict, processing_metadata: dict | None = None, session_id: str | None = None) -> int:
        """Save data with session management"""
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
                    (url, content, raw_content, analysis, processing_metadata, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (url, content, raw_content, Json(analysis), Json(processing_metadata), session_id)
                )
                self.conn.commit()
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            self.conn.rollback()
            raise

    def save_media(self, harvested_data_id: int, media_type: str, url: str, metadata: dict, session_id: str | None = None) -> int:
        """Save media content with session management"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO harvested_media 
                    (harvested_data_id, media_type, url, metadata, session_id)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (harvested_data_id, media_type, url, Json(metadata), session_id)
                )
                self.conn.commit()
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to save media: {str(e)}")
            self.conn.rollback()
            raise

    def get_all_data(self, include_raw: bool = False, include_media: bool = True, session_id: str | None = None) -> list:
        """Get all data with options for raw content and media"""
        try:
            with self.conn.cursor() as cur:
                base_query = """
                    SELECT d.id, d.url, d.content{}, d.analysis, d.processing_metadata, d.created_at{}
                    FROM harvested_data d
                    {}
                    WHERE 1=1
                    {}
                    ORDER BY d.created_at DESC
                """
                
                raw_select = ", d.raw_content" if include_raw else ""
                media_join = """
                    LEFT JOIN LATERAL (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'type', media_type,
                                'url', url,
                                'metadata', metadata
                            )
                        ) as media_data
                        FROM harvested_media
                        WHERE harvested_data_id = d.id
                    ) m ON true
                """ if include_media else ""
                media_select = ", m.media_data" if include_media else ""
                session_where = "AND d.session_id = %s" if session_id else ""

                query = base_query.format(raw_select, media_select, media_join, session_where)
                
                if session_id:
                    cur.execute(query, (session_id,))
                else:
                    cur.execute(query)
                    
                return cur.fetchall() or []
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            raise

    def mark_permanent(self, data_id: int, session_id: str) -> None:
        """Mark data and associated media as permanent"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE harvested_data 
                    SET is_temporary = FALSE 
                    WHERE id = %s AND session_id = %s;
                    
                    UPDATE harvested_media 
                    SET is_temporary = FALSE 
                    WHERE harvested_data_id = %s AND session_id = %s;
                    """,
                    (data_id, session_id, data_id, session_id)
                )
                self.conn.commit()
                logger.info(f"Marked data_id {data_id} as permanent")
        except Exception as e:
            logger.error(f"Failed to mark data as permanent: {str(e)}")
            self.conn.rollback()
            raise

    def cleanup_temporary_data(self, max_age_hours: int = 24) -> None:
        """Clean up temporary data older than specified hours"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM harvested_media 
                    WHERE is_temporary = TRUE 
                    AND created_at < NOW() - interval '%s hours';
                    
                    DELETE FROM harvested_data 
                    WHERE is_temporary = TRUE 
                    AND created_at < NOW() - interval '%s hours';
                    """,
                    (max_age_hours, max_age_hours)
                )
                self.conn.commit()
                logger.info(f"Cleaned up temporary data older than {max_age_hours} hours")
        except Exception as e:
            logger.error(f"Failed to clean up temporary data: {str(e)}")
            self.conn.rollback()
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
            self.conn.rollback()
            raise

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Database connection closed")
