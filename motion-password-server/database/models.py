from sqlalchemy import LargeBinary, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base
from .enums import DistanceAlgorithmEnumDB
from sqlalchemy import Enum as SQLAlchemyEnum

"""
User Model:

id: Primary key for the user
username: Unique username for the user
created_at: Timestamp for when the user was created
embeddings: A relationship with the Embedding model, indicating which embeddings belong to this user

Model Model:

id: Primary key for the model
name: Name of the model
path: Path to the model file
version: Model version
window_size: Window size for extracting features
overlap: Overlap for extracting features
fps: Frame rate for extracting features
distance_algorithm: Enumeration for the distance algorithm used by the model
embeddings: A relationship with the Embedding model, indicating which embeddings are associated with this model

Embedding Model:

id: Primary key for the embedding
created_at: Timestamp for when the embedding was created
embedding_data: LargeBinary field for storing the embedding data
quality: Quality score for the embedding
owner_id: Foreign key referencing the id of the corresponding User
owner: A relationship back to the User model, indicating which user owns this embedding
model_id: Foreign key referencing the id of the corresponding Model
model: A relationship back to the Model model, indicating which model this embedding is associated with
"""

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    embeddings = relationship("Embedding", back_populates="owner")


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    version = Column(Integer)
    window_size = Column(Integer)
    overlap = Column(Integer)
    fps = Column(Integer)
    distance_algorithm = Column(SQLAlchemyEnum(DistanceAlgorithmEnumDB), default=DistanceAlgorithmEnumDB.cosine_similarity)
    created_at = Column(DateTime, default=datetime.utcnow)


    embeddings = relationship("Embedding", back_populates="model")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding_data = Column(LargeBinary)
    quality = Column(Integer, index=True)

    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="embeddings")

    # Foreign Key relationship with Model
    model_id = Column(Integer, ForeignKey("models.id"))
    model = relationship("Model", back_populates="embeddings")
