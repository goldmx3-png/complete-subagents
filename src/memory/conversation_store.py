"""
Conversation memory store using PostgreSQL
"""

from typing import List, Dict, Optional
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
from src.config import settings

Base = declarative_base()


class Conversation(Base):
    """Conversation model"""
    __tablename__ = "conversations"

    conversation_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    summary = Column(Text)
    meta_data = Column("metadata", JSON)  # Renamed to avoid SQLAlchemy reserved word


class Message(Base):
    """Message model"""
    __tablename__ = "messages"

    message_id = Column(String, primary_key=True)
    conversation_id = Column(String, nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_data = Column("metadata", JSON)  # Renamed to avoid SQLAlchemy reserved word
    tokens_used = Column(Integer)


class ConversationStore:
    """Store and retrieve conversations from PostgreSQL"""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize conversation store

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(
            self.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow
        )
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def create_tables(self):
        """Create database tables if they don't exist"""
        Base.metadata.create_all(self.engine)

    async def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation

        Args:
            user_id: User ID
            title: Conversation title
            conversation_id: Optional conversation ID

        Returns:
            Conversation ID
        """
        conversation_id = conversation_id or str(uuid.uuid4())

        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            meta_data={}  # Use meta_data column name
        )

        self.session.add(conversation)
        self.session.commit()

        return conversation_id

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: str,
        metadata: Optional[Dict] = None,
        tokens_used: Optional[int] = None
    ) -> str:
        """
        Save a message to a conversation

        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            user_id: User ID (for compatibility)
            metadata: Optional metadata
            tokens_used: Optional token count

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())

        message = Message(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            meta_data=metadata or {},  # Use meta_data column name
            tokens_used=tokens_used
        )

        self.session.add(message)

        # Update conversation updated_at
        conversation = self.session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()

        if conversation:
            conversation.updated_at = datetime.utcnow()

        self.session.commit()

        return message_id

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history

        Args:
            conversation_id: Conversation ID
            limit: Optional message limit

        Returns:
            List of messages in the format [{"role": "user", "content": "..."}]
        """
        query = self.session.query(Message).filter_by(
            conversation_id=conversation_id
        ).order_by(Message.created_at.desc())

        if limit:
            query = query.limit(limit)

        messages = query.all()

        # Reverse to chronological order and return in simple format
        messages = list(reversed(messages))

        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]

    async def get_conversation_history_full(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get full conversation history with metadata

        Args:
            conversation_id: Conversation ID
            limit: Optional message limit

        Returns:
            List of messages with full details
        """
        query = self.session.query(Message).filter_by(
            conversation_id=conversation_id
        ).order_by(Message.created_at)

        if limit:
            query = query.limit(limit)

        messages = query.all()

        return [
            {
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "metadata": msg.meta_data or {},  # Use meta_data column
                "tokens_used": msg.tokens_used
            }
            for msg in messages
        ]

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get all conversations for a user

        Args:
            user_id: User ID
            limit: Maximum conversations to return

        Returns:
            List of conversations
        """
        conversations = self.session.query(Conversation).filter_by(
            user_id=user_id
        ).order_by(Conversation.updated_at.desc()).limit(limit).all()

        return [
            {
                "conversation_id": conv.conversation_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                "summary": conv.summary,
                "metadata": conv.meta_data or {}  # Use meta_data column
            }
            for conv in conversations
        ]

    async def delete_conversation(self, conversation_id: str):
        """
        Delete a conversation and all its messages

        Args:
            conversation_id: Conversation ID
        """
        # Delete messages
        self.session.query(Message).filter_by(
            conversation_id=conversation_id
        ).delete()

        # Delete conversation
        self.session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).delete()

        self.session.commit()

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str
    ):
        """
        Update conversation title

        Args:
            conversation_id: Conversation ID
            title: New title
        """
        conversation = self.session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()

        if conversation:
            conversation.title = title
            conversation.updated_at = datetime.utcnow()
            self.session.commit()

    async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Get conversation details

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation dict or None
        """
        conversation = self.session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()

        if conversation:
            return {
                "conversation_id": conversation.conversation_id,
                "user_id": conversation.user_id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
                "summary": conversation.summary,
                "metadata": conversation.meta_data or {}  # Use meta_data column
            }

        return None

    def close(self):
        """Close database session"""
        self.session.close()


# Global instance
_store = None


def get_conversation_store() -> ConversationStore:
    """Get or create conversation store singleton"""
    global _store
    if _store is None:
        _store = ConversationStore()
        _store.create_tables()
    return _store
