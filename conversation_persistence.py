"""
Conversation Persistence: Save and resume conversations across sessions
"""
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConversationPersistence:
    """
    Handles saving and loading conversation state
    Enables resuming interrupted conversations
    """
    
    def __init__(self, storage_path: str = "./conversations"):
        """
        Initialize persistence manager
        
        Args:
            storage_path: Directory for storing conversations
        """
        self.storage_path = Path(storage_path)
        self._ensure_storage_directory()
        
        logger.info(f"ConversationPersistence initialized: {storage_path}")
    
    def _ensure_storage_directory(self) -> None:
        """Create storage directory if it doesn't exist"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Storage directory ready: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise
    
    def _get_filepath(self, session_id: str) -> Path:
        """
        Get filepath for session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to session file
        """
        # Sanitize session_id
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.storage_path / f"{safe_id}.json"
    
    def save_conversation(
        self,
        session_id: str,
        conversation_history: List[Dict[str, Any]],
        state_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save conversation to disk
        
        Args:
            session_id: Unique session identifier
            conversation_history: List of conversation messages
            state_data: Current state information
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            filepath = self._get_filepath(session_id)
            
            conversation_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'conversation_history': conversation_history,
                'state': state_data,
                'metadata': metadata or {},
                'version': '1.0'
            }
            
            # Write to temporary file first
            temp_filepath = filepath.with_suffix('.tmp')
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_filepath.replace(filepath)
            
            logger.info(f"Conversation saved: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation {session_id}: {e}", exc_info=True)
            return False
    
    def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation from disk
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation data or None if not found
        """
        try:
            filepath = self._get_filepath(session_id)
            
            if not filepath.exists():
                logger.debug(f"Conversation not found: {session_id}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            logger.info(f"Conversation loaded: {session_id}")
            return conversation_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in conversation {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading conversation {session_id}: {e}", exc_info=True)
            return None
    
    def delete_conversation(self, session_id: str) -> bool:
        """
        Delete conversation from disk
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted
        """
        try:
            filepath = self._get_filepath(session_id)
            
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Conversation deleted: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting conversation {session_id}: {e}")
            return False
    
    def list_conversations(
        self,
        limit: Optional[int] = None,
        sort_by_date: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all saved conversations
        
        Args:
            limit: Maximum number to return
            sort_by_date: Sort by timestamp (newest first)
            
        Returns:
            List of conversation summaries
        """
        try:
            conversations = []
            
            for filepath in self.storage_path.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    conversations.append({
                        'session_id': data.get('session_id'),
                        'timestamp': data.get('timestamp'),
                        'message_count': len(data.get('conversation_history', [])),
                        'state': data.get('state', {}).get('current_state'),
                        'filepath': str(filepath)
                    })
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
                    continue
            
            # Sort by timestamp if requested
            if sort_by_date:
                conversations.sort(
                    key=lambda x: x.get('timestamp', ''),
                    reverse=True
                )
            
            # Apply limit
            if limit:
                conversations = conversations[:limit]
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """
        Delete conversations older than specified days
        
        Args:
            days_old: Delete conversations older than this many days
            
        Returns:
            Number of conversations deleted
        """
        try:
            deleted_count = 0
            cutoff_date = datetime.now().timestamp() - (days_old * 86400)
            
            for filepath in self.storage_path.glob("*.json"):
                try:
                    # Check file modification time
                    file_mtime = filepath.stat().st_mtime
                    
                    if file_mtime < cutoff_date:
                        filepath.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old conversation: {filepath.name}")
                        
                except Exception as e:
                    logger.warning(f"Error processing {filepath}: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old conversations")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            conversations = self.list_conversations()
            
            total_size = sum(
                Path(conv['filepath']).stat().st_size 
                for conv in conversations
                if Path(conv['filepath']).exists()
            )
            
            return {
                'total_conversations': len(conversations),
                'storage_path': str(self.storage_path),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def export_conversation(
        self,
        session_id: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export conversation to human-readable format
        
        Args:
            session_id: Session identifier
            output_path: Optional output file path
            
        Returns:
            Exported text or None if error
        """
        try:
            data = self.load_conversation(session_id)
            if not data:
                return None
            
            # Format conversation
            lines = [
                f"Conversation: {session_id}",
                f"Date: {data.get('timestamp')}",
                f"State: {data.get('state', {}).get('current_state')}",
                "=" * 60,
                ""
            ]
            
            for msg in data.get('conversation_history', []):
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                lines.append(f"{role}: {content}")
                lines.append("")
            
            export_text = "\n".join(lines)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(export_text)
                logger.info(f"Conversation exported to {output_path}")
            
            return export_text
            
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            return None


class SessionManager:
    """
    Manages conversation sessions with automatic persistence
    """
    
    def __init__(
        self,
        persistence: ConversationPersistence,
        auto_save: bool = True,
        save_interval: int = 5
    ):
        """
        Initialize session manager
        
        Args:
            persistence: Persistence handler
            auto_save: Enable automatic saving
            save_interval: Save after this many new messages
        """
        self.persistence = persistence
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("SessionManager initialized")
    
    def create_session(self, session_id: str) -> bool:
        """
        Create new session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if created
        """
        try:
            if session_id in self.active_sessions:
                logger.warning(f"Session already exists: {session_id}")
                return False
            
            self.active_sessions[session_id] = {
                'conversation_history': [],
                'state': {},
                'message_count': 0,
                'last_save': datetime.now(),
                'created_at': datetime.now()
            }
            
            logger.info(f"Session created: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get active session or load from disk
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try loading from disk
        loaded_data = self.persistence.load_conversation(session_id)
        if loaded_data:
            self.active_sessions[session_id] = {
                'conversation_history': loaded_data.get('conversation_history', []),
                'state': loaded_data.get('state', {}),
                'message_count': len(loaded_data.get('conversation_history', [])),
                'last_save': datetime.now(),
                'created_at': datetime.fromisoformat(loaded_data.get('timestamp'))
            }
            logger.info(f"Session resumed: {session_id}")
            return self.active_sessions[session_id]
        
        return None
    
    def save_session(
        self,
        session_id: str,
        force: bool = False
    ) -> bool:
        """
        Save session to disk
        
        Args:
            session_id: Session identifier
            force: Force save regardless of interval
            
        Returns:
            True if saved
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            # Check if save needed
            if not force and self.auto_save:
                messages_since_save = (
                    session['message_count'] - 
                    session.get('messages_at_last_save', 0)
                )
                if messages_since_save < self.save_interval:
                    return False
            
            # Save to disk
            success = self.persistence.save_conversation(
                session_id=session_id,
                conversation_history=session['conversation_history'],
                state_data=session['state']
            )
            
            if success:
                session['last_save'] = datetime.now()
                session['messages_at_last_save'] = session['message_count']
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def close_session(self, session_id: str, save: bool = True) -> bool:
        """
        Close and optionally save session
        
        Args:
            session_id: Session identifier
            save: Save before closing
            
        Returns:
            True if closed successfully
        """
        try:
            if session_id not in self.active_sessions:
                return False
            
            if save:
                self.save_session(session_id, force=True)
            
            del self.active_sessions[session_id]
            logger.info(f"Session closed: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            return False