"""This is batch script for embedding User's slack messages.
    - easiest: using spaCy trained model
    - harder but better: train custom model with spaCy or gensim
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Final

import pandas as pd
import spacy
from dotenv import load_dotenv
from gensim.models.doc2vec import Doc2Vec
from janome.tokenizer import Tokenizer
from tqdm import tqdm

from src.config import *
from src.load import DlgDwhLoader
from src.preprocess import clean_msg
from src.utils import *


# ========================================================
# Global Objects
# ========================================================
HERE = str(Path(__file__).resolve().parent)
t = Tokenizer()


# ========================================================
# Functions
# ========================================================
def main(model_path: str):
    """word embedding for slack messages.
    """
    # ----------------------------------------------------
    # load envvar
    # ----------------------------------------------------
    load_dotenv(dotenv_path=f'{HERE}/.env')

    # ----------------------------------------------------
    # load trained model
    # ----------------------------------------------------
    # spaCy version: # nlp = spacy.load('ja_core_news_lg')
    print(f'Loading model from {model_path}')
    model = Doc2Vec.load(model_path)

    # ----------------------------------------------------
    # get users id list
    # ----------------------------------------------------
    print('Loading users data')
    loader = DlgDwhLoader(os.environ['BQ_PROJECT_ID'])
    users_mart = loader.users_mart().to_dataframe()
    users = users_mart[['user_id', 'name']]

    # ----------------------------------------------------
    # save users' data
    # ----------------------------------------------------
    with open(USERS_DATA_PATH, 'wb') as f:
        pickle.dump(users, f)
        print(f'Saved users data to {USERS_DATA_PATH}')

    # ----------------------------------------------------
    # vectorizing messages per user
    # ----------------------------------------------------
    print(f'Vectorizing messages per user (# of users: {users.shape[0]})')
    mkdir_if_not_exist(USERS_VEC_DIR)
    for (i, row) in tqdm(list(users.iterrows()), desc='[save vector]'):
        # get per user
        uid = row['user_id']
        uname = row['name']
        u_msgs = loader.msgs_by_user(user_id=uid, ch_join_msg=False).to_dataframe()[['user_id', 'text']]

        # concat all of posted messages
        u_msgs_str = ' '.join(u_msgs['text'].values.tolist())

        # remove noise
        u_msgs_str = clean_msg(u_msgs_str)

        # vectorize
        # > https://spacy.io/api/doc
        # > https://spacy.io/api/vectors
        # spaCy version: # doc = nlp(u_msgs_str)
        # spaCy version: # vector = doc.vector.tolist()
        u_msgs_str_wakati = list(t.tokenize(u_msgs_str, wakati=True))
        vector = model.infer_vector(u_msgs_str_wakati).tolist()
        with open(f'{USERS_VEC_DIR}/{uid}.json', 'w', encoding='utf-8') as f:
            json.dump(vector, f, indent=2)


if __name__ == "__main__":
    args = sys.argv
    model_path = args[1] if len(args) > 1 else DEFAULT_MODEL_PATH
    if len(args) < 2:
        print(f'[usage] {args[0]} [model_path]')
        print(f'default model_path: {DEFAULT_MODEL_PATH}')

    main(model_path)

