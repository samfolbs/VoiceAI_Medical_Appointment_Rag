"""
voice/persistence.py
Save and load conversation sessions to disk.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationPersistence:
    """File-based conversation persistence (JSON)."""

    def __init__(self, storage_path: str = "./conversations") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("ConversationPersistence ready at %s", self.storage_path)

    def _filepath(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.storage_path / f"{safe}.json"

    # ------------------------------------------------------------------
    # Public API — parameter names match VoiceAgent usage
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        session_id: str,
        conversation_history: List[Dict[str, Any]],
        state_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Persist a conversation session.

        Args:
            session_id: Unique session identifier.
            conversation_history: List of OpenAI message dicts.
            state_data: Dict containing 'state' (str) and 'collected_data' (dict).
            metadata: Optional extra metadata.
        """
        try:
            payload = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": conversation_history,
                "state": state_data,          # keyed as 'state' for compatibility
                "metadata": metadata or {},
                "version": "1.1",
            }
            tmp = self._filepath(session_id).with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            tmp.replace(self._filepath(session_id))
            logger.debug("Conversation saved: %s", session_id)
            return True
        except Exception as exc:
            logger.error("Error saving conversation %s: %s", session_id, exc, exc_info=True)
            return False

    def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a conversation session.

        Returns a dict with keys:
          - 'conversation_history'  (list)
          - 'state'                 (str)   — current state enum value
          - 'collected_data'        (dict)  — patient info gathered so far
        or None if not found.
        """
        filepath = self._filepath(session_id)
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data: Dict[str, Any] = json.load(fh)

            # Normalise old format (v1.0) where state was stored flat
            state_blob = data.get("state", {})
            if isinstance(state_blob, str):
                # legacy: state was stored as a plain string
                state_str = state_blob
                collected = data.get("collected_data", {})
            else:
                state_str = state_blob.get("current_state", state_blob.get("state", "greeting"))
                collected = state_blob.get("collected_data", {})

            return {
                "conversation_history": data.get("conversation_history", []),
                "state": state_str,
                "collected_data": collected,
            }
        except json.JSONDecodeError as exc:
            logger.error("Corrupt conversation file %s: %s", session_id, exc)
            return None
        except Exception as exc:
            logger.error("Error loading conversation %s: %s", session_id, exc, exc_info=True)
            return None

    def delete_conversation(self, session_id: str) -> bool:
        fp = self._filepath(session_id)
        if fp.exists():
            fp.unlink()
            logger.info("Conversation deleted: %s", session_id)
            return True
        return False

    def list_conversations(
        self,
        limit: Optional[int] = None,
        sort_by_date: bool = True,
    ) -> List[Dict[str, Any]]:
        convos = []
        for fp in self.storage_path.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                convos.append(
                    {
                        "session_id": d.get("session_id"),
                        "timestamp": d.get("timestamp"),
                        "message_count": len(d.get("conversation_history", [])),
                        "filepath": str(fp),
                    }
                )
            except Exception:
                continue
        if sort_by_date:
            convos.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return convos[:limit] if limit else convos

    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        cutoff = datetime.now().timestamp() - days_old * 86400
        deleted = 0
        for fp in self.storage_path.glob("*.json"):
            if fp.stat().st_mtime < cutoff:
                fp.unlink()
                deleted += 1
        if deleted:
            logger.info("Cleaned up %d old conversations", deleted)
        return deleted
