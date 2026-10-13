import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

STORAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_sessions.json"


def _ensure_storage_dir():
    STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STORAGE_PATH.exists():
        with open(STORAGE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)


def load_all_sessions() -> Dict[str, Any]:
    """Loads all chat sessions from disk."""
    _ensure_storage_dir()
    try:
        with open(STORAGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_session(session_id: str, session_data: Dict[str, Any]):
    """Saves or updates a single chat session."""
    _ensure_storage_dir()
    sessions = load_all_sessions()
    sessions[session_id] = session_data
    with open(STORAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


def delete_session(session_id: str):
    """Deletes a chat session from disk."""
    _ensure_storage_dir()
    sessions = load_all_sessions()
    if session_id in sessions:
        del sessions[session_id]
        with open(STORAGE_PATH, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)


def create_new_session(title: str = "New Chat", pdf_name: Optional[str] = None, namespace: Optional[str] = None) -> str:
    """Creates a new empty chat session and returns its ID."""
    session_id = str(uuid.uuid4())[:8]
    if not namespace:
        # Pinecone namespace names: alphanumeric and hyphens
        namespace = f"chat-{session_id}"
    
    session_data = {
        "id": session_id,
        "title": title,
        "pdf_name": pdf_name,
        "namespace": namespace,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }
    save_session(session_id, session_data)
    return session_id
