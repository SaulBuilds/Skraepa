import os
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import Json, execute_batch
import logging
import uuid
from typing import List, Dict, Any, Optional

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
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # Start transaction
            cursor.execute("BEGIN;")
            
            # Create harvested_data table with all columns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS harvested_data (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    content TEXT,
                    raw_content TEXT,
                    analysis JSONB,
                    processing_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR(64),
                    is_temporary BOOLEAN DEFAULT TRUE,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create harvested_media table with all columns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS harvested_media (
                    id SERIAL PRIMARY KEY,
                    harvested_data_id INTEGER REFERENCES harvested_data(id) ON DELETE CASCADE,
                    media_type VARCHAR(10) NOT NULL,
                    url TEXT NOT NULL,
                    metadata JSONB,
                    download_status VARCHAR(20) DEFAULT 'pending',
                    content_type VARCHAR(100),
                    file_size BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR(64),
                    is_temporary BOOLEAN DEFAULT TRUE
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_temp ON harvested_data(session_id, is_temporary);
                CREATE INDEX IF NOT EXISTS idx_media_session_temp ON harvested_media(session_id, is_temporary);
                CREATE INDEX IF NOT EXISTS idx_media_type_status ON harvested_media(media_type, download_status);
                CREATE INDEX IF NOT EXISTS idx_media_content_type ON harvested_media(content_type);
                CREATE INDEX IF NOT EXISTS idx_media_created_at ON harvested_media(created_at);
            """)
            
            # Commit transaction
            self.conn.commit()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            # Rollback transaction on error
            if self.conn:
                self.conn.rollback()
            logger.error(f"Failed to create tables: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()

    def save_batch_media(self, media_entries: List[Dict[str, Any]], session_id: Optional[str] = None) -> List[int]:
        """Save multiple media entries in batch"""
        try:
            with self.conn.cursor() as cur:
                query = """
                    INSERT INTO harvested_media 
                    (harvested_data_id, media_type, url, metadata, session_id, 
                     download_status, content_type, file_size)
                    VALUES (
                        %(harvested_data_id)s, 
                        %(media_type)s, 
                        %(url)s, 
                        %(metadata)s,
                        %(session_id)s,
                        %(download_status)s,
                        %(content_type)s,
                        %(file_size)s
                    )
                    RETURNING id
                """
                
                # Prepare batch data
                batch_data = []
                for entry in media_entries:
                    download_info = entry.get('metadata', {}).get('download_info', {})
                    batch_data.append({
                        'harvested_data_id': entry['harvested_data_id'],
                        'media_type': entry['media_type'],
                        'url': entry['url'],
                        'metadata': Json(entry['metadata']),
                        'session_id': session_id,
                        'download_status': download_info.get('status', 'pending'),
                        'content_type': download_info.get('content_type'),
                        'file_size': download_info.get('size')
                    })

                # Execute batch insert
                results = []
                for data in batch_data:
                    cur.execute(query, data)
                    results.append(cur.fetchone()[0])

                self.conn.commit()
                logger.info(f"Batch saved {len(results)} media entries")
                return results

        except Exception as e:
            logger.error(f"Failed to save batch media: {str(e)}")
            self.conn.rollback()
            raise

    def get_media_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about stored media"""
        try:
            with self.conn.cursor() as cur:
                query = """
                    SELECT 
                        media_type,
                        COUNT(*) as total_count,
                        COUNT(CASE WHEN download_status = 'available' THEN 1 END) as available_count,
                        SUM(CASE WHEN file_size IS NOT NULL THEN file_size ELSE 0 END) as total_size,
                        COUNT(DISTINCT content_type) as format_count
                    FROM harvested_media
                    WHERE ($1::varchar IS NULL OR session_id = $1)
                    GROUP BY media_type
                """
                
                cur.execute(query, (session_id,))
                results = cur.fetchall()
                
                statistics = {
                    'images': {'count': 0, 'available': 0, 'total_size': 0, 'formats': 0},
                    'videos': {'count': 0, 'available': 0, 'total_size': 0, 'formats': 0},
                    'total': {'count': 0, 'available': 0, 'total_size': 0, 'formats': 0}
                }
                
                for row in results:
                    media_type, total, available, size, formats = row
                    statistics[media_type] = {
                        'count': total,
                        'available': available,
                        'total_size': size,
                        'formats': formats
                    }
                    statistics['total']['count'] += total
                    statistics['total']['available'] += available
                    statistics['total']['total_size'] += size
                    statistics['total']['formats'] += formats
                
                return statistics
        except Exception as e:
            logger.error(f"Failed to get media statistics: {str(e)}")
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
