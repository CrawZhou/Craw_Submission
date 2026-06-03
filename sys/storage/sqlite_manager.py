import sqlite3
import os
from typing import List, Optional, Tuple, Dict, Any

from utils.logger import get_logger

logger = get_logger("SQLiteManager")


class SQLiteManager:
    def __init__(self, db_path: str = "./video_db.sqlite"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._connect()
        self._create_tables()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.info(f"Connected to SQLite: {self.db_path}")

    def _create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT UNIQUE NOT NULL,
                video_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS vsus (
                id INTEGER PRIMARY KEY,
                clip_path TEXT NOT NULL,
                source_video_id INTEGER,
                source_video_path TEXT,
                frame_start INTEGER NOT NULL,
                frame_end INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_video_id) REFERENCES videos(id)
            )
        """)
        self.conn.commit()

    def insert_video(self, video_path: str) -> int:
        video_name = os.path.basename(video_path)

        try:
            self.cursor.execute(
                "INSERT INTO videos (video_path, video_name) VALUES (?, ?)",
                (video_path, video_name)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            logger.warning(f"Video already exists: {video_path}")
            return -1

    def insert_videos_from_directory(
        self,
        directory: str,
        extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv')
    ) -> int:
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return 0

        count = 0
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                continue

            video_path = os.path.join(directory, filename)
            if self.insert_video(video_path) > 0:
                count += 1

        logger.info(f"Inserted {count} videos from {directory}")
        return count

    def get_video_by_id(self, video_id: int) -> Optional[str]:
        self.cursor.execute(
            "SELECT video_path FROM videos WHERE id = ?",
            (video_id,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_all_videos(self) -> List[Tuple[int, str]]:
        self.cursor.execute("SELECT id, video_path FROM videos ORDER BY id")
        return self.cursor.fetchall()

    def video_exists(self, video_path: str) -> bool:
        self.cursor.execute(
            "SELECT COUNT(*) FROM videos WHERE video_path = ?",
            (video_path,)
        )
        return self.cursor.fetchone()[0] > 0

    def delete_video(self, video_id: int) -> bool:
        self.cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def count(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM videos")
        return self.cursor.fetchone()[0]

    def insert_vsu(
        self,
        vsu_id: int,
        clip_path: str,
        source_video_path: str,
        frame_start: int,
        frame_end: int,
        source_video_id: int = None
    ) -> bool:
        try:
            self.cursor.execute(
                """INSERT OR REPLACE INTO vsus
                   (id, clip_path, source_video_id, source_video_path, frame_start, frame_end)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (vsu_id, clip_path, source_video_id, source_video_path, frame_start, frame_end)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to insert VSU {vsu_id}: {e}")
            return False

    def insert_vsus_from_ranges(
        self,
        msu_ranges: List[Tuple[int, int]],
        clip_dir: str,
        source_video_path: str,
        source_video_id: int = None,
        video_fps: float = 30.0
    ) -> int:
        os.makedirs(clip_dir, exist_ok=True)

        count = 0
        for vsu_id, (start, end) in enumerate(msu_ranges):
            clip_filename = f"vsu_{vsu_id:04d}_f{start}-{end}.mp4"
            clip_path = os.path.join(clip_dir, clip_filename)

            self.insert_vsu(
                vsu_id=vsu_id,
                clip_path=clip_path,
                source_video_path=source_video_path,
                frame_start=start,
                frame_end=end,
                source_video_id=source_video_id
            )
            count += 1

        logger.info(f"Inserted {count} VSU records (clip dir: {clip_dir})")
        return count

    def get_vsu_by_id(self, vsu_id: int) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            """SELECT id, clip_path, source_video_id, source_video_path,
                      frame_start, frame_end, created_at
               FROM vsus WHERE id = ?""",
            (vsu_id,)
        )
        result = self.cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'clip_path': result[1],
                'source_video_id': result[2],
                'source_video_path': result[3],
                'frame_start': result[4],
                'frame_end': result[5],
                'created_at': result[6]
            }
        return None

    def get_all_vsus(self) -> List[Dict[str, Any]]:
        self.cursor.execute(
            """SELECT id, clip_path, source_video_id, source_video_path,
                      frame_start, frame_end, created_at
               FROM vsus ORDER BY id"""
        )
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'clip_path': row[1],
                'source_video_id': row[2],
                'source_video_path': row[3],
                'frame_start': row[4],
                'frame_end': row[5],
                'created_at': row[6]
            })
        return results

    def get_vsu_clip_path(self, vsu_id: int) -> Optional[str]:
        self.cursor.execute("SELECT clip_path FROM vsus WHERE id = ?", (vsu_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def vsu_exists(self, vsu_id: int) -> bool:
        self.cursor.execute("SELECT COUNT(*) FROM vsus WHERE id = ?", (vsu_id,))
        return self.cursor.fetchone()[0] > 0

    def delete_vsu(self, vsu_id: int) -> bool:
        self.cursor.execute("DELETE FROM vsus WHERE id = ?", (vsu_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def clear_vsus(self) -> int:
        self.cursor.execute("DELETE FROM vsus")
        self.conn.commit()
        return self.cursor.rowcount

    def count_vsus(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM vsus")
        return self.cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("SQLite connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()