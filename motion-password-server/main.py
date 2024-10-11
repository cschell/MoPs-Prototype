import concurrent.futures
import logging
import os

from fastapi.responses import JSONResponse
from fastapi import Body, FastAPI, Form
from fastapi.exceptions import HTTPException
from fastapi import Depends
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from helper_functions.functions import *

from typing import List
from sqlalchemy.orm import Session

from database import models
from database.database import SessionLocal, engine
from database.schemas import *
from database.crud import *

app = FastAPI(debug=True)

models.Base.metadata.create_all(bind=engine)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dict = {}
@app.get("/")
async def read_main():
    """
    Root endpoint returning a simple message.

    Returns:
        dict: Message {"msg": "Hello World"}.
    """
    return {"msg": "Hello World"}

# Dependency to get the database session
def get_db():
    """
    Dependency to get the database session.

    Yields:
        Session: Database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/v1/user", response_model=UserResponse)
def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user.

    Args:
        user (UserCreate): User creation details.
        db (Session): Database session.

    Returns:
        UserResponse: Created user.
    """
    db_user = create_user(db, user)
    if db_user == "Username exists already!":
        raise HTTPException(status_code=400, detail="Username exists already")
    return db_user


@app.get("/v1/read_users", response_model=List[UserResponse])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get a list of users.

    Args:
        skip (int): Number of records to skip.
        limit (int): Maximum number of records to return.
        db (Session): Database session.

    Returns:
        List[UserResponse]: List of users.
    """
    users = get_users(db, skip=skip, limit=limit)
    return users


@app.get("/v1/user/{user_id}", response_model=UserResponse)
def read_user(user_id: int, db: Session = Depends(get_db)):
    """
    Get a user by ID.

    Args:
        user_id (int): User ID.
        db (Session): Database session.

    Returns:
        UserResponse: User details.
    """
    db_user = get_user(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.delete("/v1/user/{user_id}", response_model=UserResponse)
def delete_user_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    Delete a user by ID.

    Args:
        user_id (int): User ID.
        db (Session): Database session.

    Returns:
        UserResponse: Deleted user.
    """
    deleted_user = delete_user(db, user_id)

    if deleted_user:
        return deleted_user
    else:
        raise HTTPException(status_code=404, detail="User not found")
    

@app.delete("/v1/user/delete_by_username/{username}")
def delete_user_endpoint_by_username(username: str, db: Session = Depends(get_db)):
    """
    Delete a user by username.

    Args:
        username (str): Username.
        db (Session): Database session.

    Returns:
        SuccessResponse: Success message.
    """
    deleted_username = delete_user_username(db, username)

    if deleted_username:
        return username
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.get("/v1/read_embeddings", response_model=List[EmbeddingResponse])
def read_embeddings(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get a list of embeddings.

    Args:
        skip (int): Number of records to skip.
        limit (int): Maximum number of records to return.
        db (Session): Database session.

    Returns:
        list[Embedding]: List of embeddings.
    """
    try:
        embeddings = get_embeddings(db, skip=skip, limit=limit)
        return embeddings
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/embedding/{embedding_id}", response_model=EmbeddingResponse)
def get_embedding_endpoint(embedding_id: int, db: Session = Depends(get_db)):
    """
    Get an embedding by ID.

    Args:
        embedding_id (int): Embedding ID.
        db (Session): Database session.

    Returns:
        Embedding: Embedding details.
    """
    embedding = get_embedding(db, embedding_id)
    return embedding


@app.delete("/v1/embedding/{embedding_id}", response_model=EmbeddingResponse)
def delete_embedding_endpoint(embedding_id: int, db: Session = Depends(get_db)):
    """
    Delete an embedding by ID.

    Args:
        embedding_id (int): Embedding ID.
        db (Session): Database session.

    Returns:
        SuccessResponse: Success message.
    """
    deleted_embedding = delete_embedding(db, embedding_id)

    if deleted_embedding:
        return deleted_embedding
    else:
        raise HTTPException(status_code=404, detail="Embedding not found")


@app.delete("/v1/all_embeddings/user/{user_id}", response_model=List[EmbeddingResponse])
def delete_all_embedding_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    Delete all embeddings for a user.

    Args:
        user_id (int): User ID.
        db (Session): Database session.

    Returns:
        List[EmbeddingResponse]: List of deleted embeddings.
    """
    deleted_embeddings = delete_all_embedding_from_user(db, user_id)

    if deleted_embeddings:
        return deleted_embeddings
    else:
        raise HTTPException(status_code=404, detail="No embeddings found for the user")


@app.get("/v1/user/{user_id}/embedding", response_model=list[EmbeddingResponse])
def read_user_embeddings(user_id: int, db: Session = Depends(get_db)):
    """
    Get a list of embeddings for a user.

    Args:
        user_id (int): User ID.
        db (Session): Database session.

    Returns:
        list[Embedding]: List of embeddings.
    """
    db_user = get_user(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_embeddings = get_user_embeddings(db, user_id)
    return user_embeddings


@app.get("/v1/model", response_model=list[ModelResponse])
def read_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get a list of models.

    Args:
        skip (int): Number of records to skip.
        limit (int): Maximum number of records to return.
        db (Session): Database session.

    Returns:
        list[Model]: List of models.
    """
    models = get_models(db, skip=skip, limit=limit)
    return models


@app.get("/v1/model/{model_id}", response_model=ModelResponse)
def read_model(model_id: int, db: Session = Depends(get_db)):

    """
    Get a model by ID.

    Args:
        model_id (int): Model ID.
        db (Session): Database session.

    Returns:
        Model: Model details.
    """
    db_model = get_model(db, model_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model


@app.post("/v1/model", response_model=ModelResponse)
async def create_model_endpoint(file: UploadFile, model: ModelCreate = Depends(ModelCreate), db: Session = Depends(get_db)
):
    """
    Create a new model.

    Args:
        model (ModelCreate): Model creation details.
        model_file (UploadFile): Uploaded file.
        db (Session): Database session.

    Returns:
        Model: Created model.
    """
    try:
        logger.info(f"Received file: {file.filename}")
        file_path = os.path.join('models', file.filename)

        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        logger.info(f"File saved to: {file_path}")

        db_model = create_model(db, model)

        logger.info(f"Model created: {db_model}")
        
        return db_model
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


@app.delete("/v1/model/{model_id}", response_model=ModelResponse)
def delete_model_endpoint(model_id: int, db: Session = Depends(get_db)):
    """
    Delete a model by ID.

    Args:
        model_id (int): Model ID.
        db (Session): Database session.

    Returns:
        SuccessResponse: Success message.
    """
    deleted_model = delete_model(db, model_id)

    if deleted_model:
        return deleted_model
    else:
        raise HTTPException(status_code=404, detail="Model not found")



@app.post("/v2/enrollment/{user_id}")
async def enrollment_user_motion_data(file: UploadFile, user_id: int, model_id: int, db: Session = Depends(get_db)):
    """
    Upload motion data for user enrollment.

    Args:
        file (UploadFile): The motion data file (CSV or GZ format).
        user_id (int): The ID of the user for enrollment.
        model_id (int): The ID of the motion model.
        db (Session): Database session.

    Returns:
        Embedding: Created embedding for the user.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and GZ files are allowed.")

        result, model_information = prepare_data(file, model_id, model_dict, db)

        embedding_bytes = code_to_json_byte(result)

        db_embedding = create_embedding(db=db, quality=0, user_id=user_id, model_id=model_id, embedding_data=embedding_bytes)
        
        if db_embedding:
            return JSONResponse(content={"message": "Enrollment successful", "user id": user_id, "embedding id": db_embedding.id})
        else:
            return JSONResponse(content={"message": "Enrollment failed", "user id": user_id, "embedding id": db_embedding.id})
    except Exception as e:
        logging.error("%s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/authentication/{user_id}")
async def authentication_user_motion_data(file: UploadFile, user_id: int, model_id: int, threshold: float, db: Session = Depends(get_db)):
    """
    Compare motion data for user authentication.

    Args:
        file (UploadFile): The motion data file (CSV or GZ format).
        user_id (int): The ID of the user for authentication.
        model_id (int): The ID of the motion model.
        db (Session): Database session.

    Returns:
        float: Probability of the user's authentication.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and GZ files are allowed.")
        print("hallo")

        result, model_information = prepare_data(file, model_id, model_dict, db)
        print("finish")
        reference_user_embeddings = get_embeddings_by_user_and_model(db, user_id, model_id)

        embedding_tensors = []

        # Iterate over each embedding
        for embedding in reference_user_embeddings:
            # Decode the embedding data to PyTorch tensor
            embedding_data = decode_to_pytorch_tensor(embedding.embedding_data)
            # Append the tensor to the list
            embedding_tensors.append(embedding_data)

        # Stack the list of tensors along a new dimension to create a single tensor
        stacked_embeddings = torch.stack(embedding_tensors)

        # Calculate the mean along the specified dimension (axis)
        mean_embedding = torch.mean(stacked_embeddings, dim=0)

        distance_value = compare_embeddings(result, mean_embedding, model_information.distance_algorithm)

        if distance_value >= threshold:
            return JSONResponse(content={"message": "Authentication Successful", "distance": round(distance_value, 2), "threshold": threshold})
        else:
            return JSONResponse(content={"message": "Authentication Failed", "distance": round(distance_value, 2), "threshold": threshold})
    except Exception as e:
        logging.error("%s", e)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/v2/identification")
async def identify_user_motion_data(file: UploadFile, model_id: int, db: Session = Depends(get_db)):
    """
    Identify users based on motion data.

    Args:
        file (UploadFile): The motion data file (CSV or GZ format).
        model_id (int): The ID of the motion model.
        db (Session): Database session.

    Returns:
        List[int]: List of top 5 user IDs with the highest match probability.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and GZ files are allowed.")

        query_embedding, model_information = prepare_data(file, model_id, model_dict, db)

        queried_user_embeddings = get_embeddings_by_model(db, model_id)

        all_distance_values = []

        # Iterate through all embeddings
        for embedding in queried_user_embeddings:

            embedding_data = decode_to_pytorch_tensor(embedding.embedding_data)

            distance_value = compare_embeddings(query_embedding, embedding_data, model_information.distance_algorithm)

            all_distance_values.append(("distance: " + str(round(distance_value, 2)), "username: " + embedding.owner.username))

        top_matches = sorted(all_distance_values, key=lambda x: x[0], reverse=True)[:5]

        return top_matches
    except Exception as e:
        logging.error("%s", e)
        raise HTTPException(status_code=500, detail=str(e))
