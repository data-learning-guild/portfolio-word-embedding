"""Shared Constants and Parameters"""
from typing import Final


# ========================================================
# Constants
# ========================================================
DATASET_DIR: Final[str] = './data'

DEFAULT_MODEL_PATH: Final[str] = f'{DATASET_DIR}/trained_doc2vec.model.pkl'
DEFAULT_TRAIN_DATASET_PATH: Final[str] = f'{DATASET_DIR}/train_dataset.json'
DEFAULT_TEST_DATASET_PATH: Final[str] = f'{DATASET_DIR}/test_dataset.json'

USERS_DATA_PATH: Final[str] = f'{DATASET_DIR}/users.csv.pkl'
USERS_VEC_DIR: Final[str] = f'{DATASET_DIR}/users_vec'
USERS_VEC_PATH_WILDCARD: Final[str] = '*.json'
