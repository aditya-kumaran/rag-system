import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Message:
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


class SessionState:
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        max_recent_messages: int = 10,
    ):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages: list[Message] = []
        self.papers_discussed: list[str] = []
        self.entities: dict[str, str] = {}
        self.compressed_context: str = ""
        self.max_recent_messages = max_recent_messages
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.updated_at = datetime.utcnow().isoformat()

    def add_papers_discussed(self, arxiv_ids: list[str]) -> None:
        for aid in arxiv_ids:
            if aid and aid not in self.papers_discussed:
                self.papers_discussed.append(aid)

    def add_entity(self, short_form: str, full_form: str) -> None:
        self.entities[short_form] = full_form

    def get_recent_messages(self, n: Optional[int] = None) -> list[Message]:
        n = n or self.max_recent_messages
        return self.messages[-n:]

    def get_conversation_text(self, n: Optional[int] = None) -> str:
        messages = self.get_recent_messages(n)
        parts = []
        for msg in messages:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "messages": [m.to_dict() for m in self.messages],
            "papers_discussed": self.papers_discussed,
            "entities": self.entities,
            "compressed_context": self.compressed_context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        session = cls(conversation_id=data.get("conversation_id"))
        session.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        session.papers_discussed = data.get("papers_discussed", [])
        session.entities = data.get("entities", {})
        session.compressed_context = data.get("compressed_context", "")
        session.created_at = data.get("created_at", "")
        session.updated_at = data.get("updated_at", "")
        return session


class SessionManager:
    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._sessions: dict[str, SessionState] = {}

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    conversation_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def create_session(self) -> SessionState:
        session = SessionState()
        self._sessions[session.conversation_id] = session
        self._save_session(session)
        return session

    def get_session(self, conversation_id: str) -> Optional[SessionState]:
        if conversation_id in self._sessions:
            return self._sessions[conversation_id]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM sessions WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = cursor.fetchone()

        if row:
            data = json.loads(row[0])
            session = SessionState.from_dict(data)
            self._sessions[conversation_id] = session
            return session

        return None

    def save_session(self, session: SessionState) -> None:
        self._sessions[session.conversation_id] = session
        self._save_session(session)

    def _save_session(self, session: SessionState) -> None:
        data = json.dumps(session.to_dict(), default=str)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (conversation_id, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    session.conversation_id,
                    data,
                    session.created_at,
                    session.updated_at,
                ),
            )
            conn.commit()

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT conversation_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            {"conversation_id": r[0], "created_at": r[1], "updated_at": r[2]}
            for r in rows
        ]

    def delete_session(self, conversation_id: str) -> None:
        self._sessions.pop(conversation_id, None)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.commit()
