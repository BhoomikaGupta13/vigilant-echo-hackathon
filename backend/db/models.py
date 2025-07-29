# backend/db/models.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from .database import Base

class Source(Base):
    """
    SQLAlchemy model for tracking misinformation sources.
    
    This table stores information about sources (like @TwitterHandle, Website.com)
    and tracks how many times content from each source has been flagged by our AI system.
    """
    __tablename__ = "sources"  # This will be the name of the table in the database
    
    # Primary key - auto-incrementing integer ID
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Source identifier (e.g., @TwitterHandle, Website.com) - must be unique
    source_id = Column(String, unique=True, index=True, nullable=False)
    
    # Number of times this source has been flagged for misinformation
    flag_count = Column(Integer, default=0, nullable=False)
    
    # Boolean flag indicating if this source is considered high risk
    # (True if flag_count reaches a certain threshold)
    is_high_risk = Column(Boolean, default=False, nullable=False)
    
    # Timestamp when this source record was first created
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Timestamp when this source was last flagged (nullable - might never be flagged)
    last_flagged_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        """String representation of the Source object for debugging."""
        return f"<Source(source_id='{self.source_id}', flag_count={self.flag_count}, is_high_risk={self.is_high_risk})>"