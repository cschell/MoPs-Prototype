from pydantic import BaseModel, ConfigDict
from typing import List
from datetime import datetime

from sqlalchemy import LargeBinary
from .enums import DistanceAlgorithmEnumDB


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    pass


class UserDelete(UserBase):
    pass


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime

    class Config():
        from_attributes = True


class User(UserBase):
    id: int
    created_at: datetime

    class Config():
        from_attributes = True


class EmbeddingBase(BaseModel):
    quality: int


class EmbeddingCreate(EmbeddingBase):
    pass


class EmbeddingDelete(EmbeddingBase):
    pass


class EmbeddingResponse(EmbeddingBase):
    id: int
    owner_id: int
    created_at: datetime
    model_id: int

    class Config():
        from_attributes = True

class Embedding(EmbeddingBase):
    id: int
    owner_id: int
    created_at: datetime  

    class Config():
        from_attributes = True


class ModelBase(BaseModel):
    id: int
    name: str
    version: int  
    window_size: int
    overlap: int
    fps: int
    distance_algorithm: DistanceAlgorithmEnumDB

    class Config():
        protected_namespaces = ()
        

class ModelResponse(ModelBase):
    pass


class ModelCreate(ModelBase):
    pass


class ModelDelete(ModelBase):
    pass

