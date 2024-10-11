from sqlalchemy import LargeBinary
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload

from database.models import User, Embedding
from database.schemas import UserCreate, UserDelete, EmbeddingCreate, EmbeddingDelete
from datetime import datetime
from database.models import Model
from database.schemas import ModelCreate


def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise ValueError("Username already exists!")
    
    db_user = User(username=user.username, created_at=datetime.utcnow())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    
    if user:
        deleted_user = user

        delete_all_embedding_from_user(db=db, user_id=user_id)

        db.delete(user)
        db.commit()

        return deleted_user

    return None


def delete_user_username(db: Session, username: str):
    user = db.query(User).filter(User.username == username).first()
    if user:
        user_id = user.id
        deleted_user = user
        
        delete_all_embedding_from_user(db=db, user_id=user_id)

        db.query(User).filter(User.id == user_id).delete()
        db.commit()

        return deleted_user

    return None

def get_embedding(db: Session, embedding_id: int):
    return db.query(Embedding).filter(Embedding.id == embedding_id).first()


def get_embeddings(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Embedding).offset(skip).limit(limit).all()


def create_embedding(db: Session, quality: int, user_id: int, model_id: int, embedding_data: LargeBinary):
    user = db.query(User).filter(User.id == user_id).first()
    model = db.query(Model).filter(Model.id == model_id).first()
    if not user:
        raise ValueError("User does not exist!")
    
    if not model:
        raise ValueError("Model does not exist!")
    
    db_embedding = Embedding(
        quality=quality,
        owner_id=user_id,
        embedding_data=embedding_data,
        model_id=model_id,
        created_at=datetime.utcnow()
    )
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding



def delete_embedding(db: Session, embedding_id: int):
    embedding = db.query(Embedding).filter(Embedding.id == embedding_id).first()
    
    if embedding:
        deleted_embedding = embedding

        db.delete(embedding)
        db.commit()

        return deleted_embedding

    return None


def delete_all_embedding_from_user(db: Session, user_id: int):
    user = db.query(User).options(joinedload(User.embeddings)).filter(User.id == user_id).first()
    
    if user:
        embeddings_to_delete = user.embeddings
        db.query(Embedding).filter(Embedding.owner_id == user_id).delete()
        db.commit()
        return embeddings_to_delete
    return None


def get_user_embeddings(db: Session, user_id: int):
    return db.query(Embedding).filter(Embedding.owner_id == user_id).all()


def get_embeddings_by_user_and_model(db: Session, user_id: int, model_id: int):
    return db.query(Embedding).filter(
        Embedding.owner_id == user_id,
        Embedding.model_id == model_id
    ).all()


def get_embeddings_by_model(db: Session, model_id: int):
    return db.query(Embedding).filter(
        Embedding.model_id == model_id
    ).all()


def get_model(db: Session, model_id: int):
    return db.query(Model).filter(Model.id == model_id).first()


def get_models(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Model).offset(skip).limit(limit).all()


def create_model(db: Session, model: ModelCreate):
    db_model = Model(
        name=model.name,
        version=model.version,
        window_size=model.window_size,
        overlap=model.overlap,
        fps=model.fps,
        distance_algorithm=model.distance_algorithm,
        created_at=datetime.utcnow()
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def delete_model(db: Session, model_id: int):
    model = db.query(Model).filter(Model.id == model_id).first()
    if model:
        deleted_model = model

        db.query(Embedding).filter(Embedding.model_id == model_id).delete()
        db.query(Model).filter(Model.id == model_id).delete()
        db.commit()
        
        return deleted_model
    return None