import gzip
import io
import json
import logging
import torch
import os
import pandas as pd
import motion_learning_toolbox as mlt
import numpy as np
import concurrent.futures
from database.crud import get_model
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance, BaseDistance

from src.similarity_learning import SimilarityLearning

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
DEVICE = "cpu"
ALLOWED_EXTENSIONS = {'csv', 'gz'}

coordinate_system = {
    "forward": "z",
    "right": "x",
    "up": "y",
}

joint_names = ["hmd", "left_controller", "right_controller"]

target_joints = ["left_controller", "right_controller"]


def load_model(model_information):
    model_path = os.path.join("models", model_information.name)
    try:
        if os.path.exists(model_path):
            model = torch.jit.load(model_path, map_location=torch.device("cpu"))
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        logging.error("%s", e)
        raise


def load_model_v2(model_information):
    model_path = os.path.join("models", model_information.name)
    try:
        if os.path.exists(model_path):
            try:
                model = SimilarityLearning.load_from_checkpoint(model_path, map_location=DEVICE).eval()
            except Exception as e:
                print("Error loading model:", e)
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        logging.error("%s", e)
        raise


def _convert_coord_system_from_RUB_to_RUF(df):
    df = df.copy()

    for c in df.columns:
        if c.endswith("_z") or c.endswith("_w"):
            df[c] *= -1

    return df


def _convert_m_to_cm(df):
    df = df.copy()

    for c in df.columns:
        if "_pos_" in c:
            df[c] *= 100

    return df


def cut_data_before_gripButtonPressed_and_last_second(input_data):
    """
    This function removes data before the first grip button press (left or right)
    and removes the last 60 rows from the input data.

    Args:
        input_data: A list of strings representing the CSV rows.

    Returns:
        A list of strings representing the filtered CSV rows.
    """

    # Check for grip button press in either left or right controller column
    grip_pressed = input_data["right_controller_Grip_button_"] | input_data["left_controller_Grip_button_"]

    # Get the index of the first grip press
    first_grip_index = grip_pressed.idxmax() if grip_pressed.any() else None

    # Filter data based on grip press index and remove last 60 rows
    return input_data[first_grip_index:].iloc[:-60]


def preprocess_data(input_data, model_information):
    logging.info("Current working directory: %s", os.getcwd())

    input_data = cut_data_before_gripButtonPressed_and_last_second(input_data)

    input_data = (input_data
                  .drop(['Timestamp', 'Frame'], axis=1)
                  .pipe(_convert_coord_system_from_RUB_to_RUF)
                  .pipe(_convert_m_to_cm)
                  .assign(
        RealtimeSinceStartup=lambda df: df['RealtimeSinceStartup'] - df['RealtimeSinceStartup'].iloc[0])
                  .assign(RealtimeSinceStartup=lambda df: pd.to_timedelta(df['RealtimeSinceStartup'], unit="s"))
                  .set_index('RealtimeSinceStartup')
                  )

    resampled_data = mlt.resample(target_fps=model_information.fps, data=input_data, joint_names=joint_names)

    br_data = mlt.to_body_relative(resampled_data,
                                   target_joints=target_joints,
                                   reference_joint="hmd",
                                   coordinate_system=coordinate_system)

    bra_data = mlt.to_acceleration(br_data)

    bra_data = bra_data.dropna().sort_index(axis=1)

    return bra_data


def predict_data(model, data, model_information):
    # Ensure the model is in evaluation mode
    model.eval()

    # Parameters
    sequence_length = model_information.window_size
    overlap = sequence_length - 1000  # Set overlap to sequence_length - 1 for an overlap of 1499

    # Calculate the number of sequences you can create with the specified overlap
    num_sequences = (len(data) - overlap) // (sequence_length - overlap)

    # Initialize the tensor to store the sequences
    reshaped_dataset = torch.zeros(num_sequences, sequence_length, data.shape[1])

    # Create sequences with the specified overlap
    for i in range(num_sequences):
        start_idx = i * (sequence_length - overlap)
        end_idx = start_idx + sequence_length
        reshaped_dataset[i, :, :] = torch.tensor(data.iloc[start_idx:end_idx, :].values)
    # Display the shape of the reshaped dataset

    reshaped_array = reshaped_dataset.numpy()  # Convert PyTorch tensor to NumPy array

    # Check data types
    if not np.issubdtype(reshaped_array.dtype, np.number):
        raise ValueError("The array contains non-numeric data types.")

    # Check for NaN or infinity values
    if np.isnan(reshaped_array).any() or np.isinf(reshaped_array).any():
        raise ValueError("The array contains NaN or infinity values.")

    # Convert to PyTorch Tensor
    new_data_tensor = torch.tensor(reshaped_array, dtype=torch.float32)

    with torch.inference_mode():
        ref_embeddings = model(new_data_tensor)

    return ref_embeddings


def predict_data_v2(model, data):
    scaling = model.scaling_params

    def scale(X):
        return (X - scaling["mean"]) / scaling["std"]

    X = scale(data)

    X_values = X.values.astype(float)
    X_tensor = torch.FloatTensor(X_values)
    X_tensor = torch.unsqueeze(X_tensor, dim=0)

    # X_tensor = torch.FloatTensor([X.values.astype(float)])

    word_length = len(X)
    lengths = torch.tensor([word_length]).to(DEVICE)

    ref_embeddings = model(
        X_tensor,
        lengths=lengths,
    )

    return ref_embeddings.detach()


def code_to_json_byte(result):
    # Convert the PyTorch tensor to a serialized JSON string
    serialized_embeddings = json.dumps(result.tolist())

    # Convert the serialized JSON string to bytes
    embedding_bytes = serialized_embeddings.encode('utf-8')
    return embedding_bytes


def decode_to_pytorch_tensor(embedding_bytes):
    # Decode the byte data to get the serialized JSON string
    serialized_embeddings_back = embedding_bytes.decode('utf-8')

    # Load the JSON string into a Python list
    result_list = json.loads(serialized_embeddings_back)

    # Convert the list to a PyTorch tensor
    loaded_result = torch.tensor(result_list)
    return loaded_result


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compare_embeddings(embeddings, ref_embeddings, distance_algorithm):
    if (distance_algorithm == "cosine_similarity"):
        distance_func = CosineSimilarity()
    elif (distance_algorithm == "lpdistance"):
        distance_func = LpDistance()
    else:
        distance_func = BaseDistance()

    ref_distances = distance_func(
        embeddings,
        ref_embeddings,
    )

    return float(ref_distances.numpy()[0, 0])


def load_df(file):
    if file.filename.endswith('.gz'):
        with gzip.open(io.BytesIO(file.file.read()), 'rt') as f:
            df = pd.read_csv(f, delimiter=',', on_bad_lines="skip")
    elif file.filename.endswith('.csv'):
        with io.BytesIO(file.file.read()) as f:
            df = pd.read_csv(f, delimiter=',', on_bad_lines="skip")
    return df


def prepare_data(file, model_id, model_dict, db):
    df = load_df(file)
    model_information = get_model(db, model_id)

    if model_id in model_dict:
        model = model_dict[model_id]
        data = preprocess_data(df, model_information)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_model = executor.submit(load_model_v2, model_information)
            future_data = executor.submit(preprocess_data, df, model_information)

        model = future_model.result()
        data = future_data.result()
        model_dict[model_id] = model

    return predict_data_v2(model, data), model_information
