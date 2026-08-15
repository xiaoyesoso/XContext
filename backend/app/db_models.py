from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.core.database import Base


class ContextItemRecord(Base):
    """SQLAlchemy record for a context item."""

    __tablename__ = "context_items"

    id = Column(String, primary_key=True)
    session_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    authority = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    priority = Column(Integer, nullable=False)
    token_cost = Column(Integer, nullable=False)
    layer = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    profile_dimension = Column(String, nullable=True)
    profile_tier = Column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<ContextItemRecord id={self.id} session={self.session_id} type={self.type}>"
