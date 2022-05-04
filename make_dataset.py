"""Make dataset
    - load slack data from bigquery
    - preprocess (remove noise, morphological analysis...)
    - save it
"""
import json
import os
import random
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from janome.tokenizer import Tokenizer
from tqdm import tqdm

from src.config import *
from src.load import DlgDwhLoader
from src.preprocess import clean_msg
from src.utils import *


# ============================
# Global variables
# ============================
HERE = str(Path(__file__).resolve().parent)
t = Tokenizer()


# ============================
# Functions
# ============================
def load_slack_posts():
    """Load slack posts from bigquery
        Returns:
            list of sentence (ndarray Nx1)
    """
    # get users id list
    loader = DlgDwhLoader(os.getenv('BQ_PROJECT_ID', ''))
    users_mart = loader.users_mart().to_dataframe()
    users = users_mart[['user_id', 'name']]

    # list messages per post
    posts = np.empty(0, dtype=str)
    for (i, row) in tqdm(list(users.iterrows()), desc='[load text]'):
        # get per user
        uuid = row['user_id']
        u_msgs = loader.msgs_by_user(user_id=uuid, ch_join_msg=False).to_dataframe()['text']

        if u_msgs.shape[0] == 0:
            continue

        posts = np.hstack([posts, u_msgs.values])

    return posts

def to_dataset(posts: list=None) -> list:
    dataset = None
    for tag, post in tqdm(enumerate(posts), desc='[save dataset]'):
        # Preparation - Remove noise
        cleaned_ = clean_msg(post)
        # Preparation - Tokenize
        corpus = list(t.tokenize(cleaned_, wakati=True))
        # Add to list
        if dataset is None:
            dataset = [{"tag": tag, "text": corpus}]
        else:
            dataset.append({"tag": tag, "text": corpus})
    return dataset


def main():
    """Main proc
        Dataset:
        [
            {"tag": 0, "text": ["word", "word", ...]},
            ...
        ]
    """
    # Load envvars
    load_dotenv(os.path.join(HERE, '.env'))

    # Load slack messages
    posts = load_slack_posts()

    # Split for train(80%) and test(20%)
    random.shuffle(posts)
    train_size = int(len(posts) * 0.8)
    posts_train = posts[:train_size]
    posts_test = posts[train_size:]

    # Make dataset
    train_data = to_dataset(posts_train)
    test_data = to_dataset(posts_test)

    # Save
    mkdir_if_not_exist(DATASET_DIR)
    with open(DEFAULT_TRAIN_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(DEFAULT_TEST_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    main()
