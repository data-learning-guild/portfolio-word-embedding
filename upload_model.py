"""Upload model and user's vector files to Cloud Storage
"""
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Final

from google.cloud import storage


# ========================================================
# Constants
# ========================================================
DATASET_DIR: Final[str] = './data'
USERS_DATA_PATH: Final[str] = f'{DATASET_DIR}/users.csv.pkl'
DEFAULT_MODEL_PATH: Final[str] = f'{DATASET_DIR}/trained_doc2vec.model.pkl'
USERS_VEC_DIR: Final[str] = f'{DATASET_DIR}/users_vec'
USERS_VEC_PATH_WILDCARD: Final[str] = '*.json'

# ========================================================
# Functions
# ========================================================
def create_bucket_class_location(
    bucket_name: str, project_id: str, storage_class: str='STANDARD', location: str='asia-northeast1'):
    """Create a new bucket in specific location with storage class"""
    # bucket_name = "your-new-bucket-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    if bucket.exists(storage_client):
        print('bucket has already existed.')
        return
    bucket.storage_class = storage_class
    new_bucket = storage_client.create_bucket(bucket, location=location)

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )
    return new_bucket



def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def main(model_path: str, users_data_path: str):
    """Main proc
    """
    # ----------------------------------------------------
    # load envvar
    # ----------------------------------------------------
    load_dotenv(dotenv_path='.env')

    # ----------------------------------------------------
    # create bucket on cloud storage
    # ----------------------------------------------------
    bucket_name = os.environ['GAE_DEFAULT_BUCKET_NAME']
    project_id = os.environ['BQ_PROJECT_ID']
    create_bucket_class_location(bucket_name, project_id)

    # ----------------------------------------------------
    # upload trained model
    # ----------------------------------------------------
    p_model = Path(model_path)
    if p_model.exists():
        upload_blob(bucket_name, str(p_model), p_model.name)
    else:
        print(f'No trained model found at {model_path}')

    # ----------------------------------------------------
    # upload users table
    # ----------------------------------------------------
    p_users_tbl = Path(users_data_path)
    if p_users_tbl.exists():
        upload_blob(bucket_name, str(p_users_tbl), p_users_tbl.name)
    else:
        print(f'No users data found at {users_data_path}')

    # ----------------------------------------------------
    # upload users' vector
    # ----------------------------------------------------
    p_list = list(Path(USERS_VEC_DIR).glob(USERS_VEC_PATH_WILDCARD))
    p_list = [x for x in p_list if x.stem not in ['dataset', 'test_dataset', 'train_dataset']]
    p_vectors = p_list

    for p_vector in p_vectors:
        upload_blob(bucket_name, str(p_vector), 'vectors/' + p_vector.name)
    print(f'Uploaded {len(p_vectors)} vectors')


if __name__ == "__main__":
    args = sys.argv
    model_path = args[1] if len(args) > 1 else DEFAULT_MODEL_PATH
    users_data_path = args[2] if len(args) > 2 else USERS_DATA_PATH
    if len(args) != 3:
        print(f'Usage: {args[0]} [option:model_path] [option:users_data_path]')
        print(f'Default Model Path: {DEFAULT_MODEL_PATH}')
        print(f'Default Users Data Path: {USERS_DATA_PATH}')

    main(model_path=model_path, users_data_path=users_data_path)
