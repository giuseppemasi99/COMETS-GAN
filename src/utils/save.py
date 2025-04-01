import numpy as np
import os


def save(inference_data_path: str, file_name: str, synthetic: np.ndarray) -> None:
    os.makedirs(inference_data_path, exist_ok=True)
    with open(f'{inference_data_path}/{file_name}.npy', 'wb') as f:
        np.save(f, synthetic)
