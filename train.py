import json
import pickle
import smart_open
import sys
from typing import Final

import numpy as np
from gensim.models.doc2vec import (
    Doc2Vec, TaggedDocument)

from make_dataset import TRAIN_DATASET_PATH


# =========================
# Constants
# =========================
DATASET_DIR: Final[str] = './data'
DEFAULT_TRAIN_DATASET_PATH: Final[str] = f'{DATASET_DIR}/train_dataset.json'
DEFAULT_MODEL_PATH: Final[str] = f'{DATASET_DIR}/trained_doc2vec.model.pkl'

# Hyper parameters
HP_VECTOR_SIZE: Final[int] = 400
HP_WINDOW_SIZE: Final[int] = 8
HP_MIN_COUNT: Final[int] = 1
HP_EPOCHS: Final[int] = 40


# =========================
# Functions
# =========================
def read_corpus(fname: str, tokens_only: bool=False):
    """Read documents from a file and return a list of TaggedDocument objects
    """
    with smart_open.open(fname, encoding="utf-8") as f:
        dataset = json.load(f)
        for doc in dataset:
            doc_tag = int(doc['tag'])
            doc_text = doc['text']
            if tokens_only:
                yield doc_text
            else:
                # For training data, add tags
                yield TaggedDocument(doc_text, [doc_tag])

def cos_similarity(_x: list, _y: list) -> float:
    """compute cos similarity"""
    vx = np.array(_x)
    vy = np.array(_y)
    return np.dot(vx, vy) / (np.linalg.norm(vx) * np.linalg.norm(vy))


def simulate_matching(model: Doc2Vec):
    """simple testing for matching with trained model"""
    v_py1_ja4 = model.infer_vector(['python', 'java', 'java', 'java', 'java'])
    v_py3_ja2 = model.infer_vector(['python', 'python', 'python', 'java', 'java'])
    v_py4_ja1 = model.infer_vector(['python', 'python', 'python', 'python', 'java'])
    v_py5_ja0 = model.infer_vector(['python', 'python', 'python', 'python', 'python'])

    print('===============================')
    print('「python x 5」文ベクトルとのCOS類似度')
    print('-------------------------------')
    print(f'python x 1, java x 4 >> {cos_similarity(v_py5_ja0, v_py1_ja4)}')
    print(f'python x 3, java x 2 >> {cos_similarity(v_py5_ja0, v_py3_ja2)}')
    print(f'python x 4, java x 1 >> {cos_similarity(v_py5_ja0, v_py4_ja1)}')
    print('===============================')


# =========================
# Main
# =========================
def main(dataset_path: str, model_path: str):
    """Main function"""
    # --------------------------------------------------
    # Load train dataset
    # --------------------------------------------------
    train_copus = list(read_corpus(fname=dataset_path))

    print(f'# of train corpus: {len(train_copus)} is loaded.')

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    model = Doc2Vec(
        vector_size=HP_VECTOR_SIZE,
        window=HP_WINDOW_SIZE,
        min_count=HP_MIN_COUNT,
        epochs=HP_EPOCHS)

    model.build_vocab(train_copus)
    print('start training...')
    model.train(train_copus, total_examples=model.corpus_count, epochs=model.epochs)
    print('finished training.')

    # --------------------------------------------------
    # Simulate matching with trained model
    # --------------------------------------------------
    simulate_matching(model)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        print(f'Trained model is saved to {model_path}')


if __name__ == "__main__":
    args = sys.argv
    dataset_path = args[1] if len(args) > 1 else DEFAULT_TRAIN_DATASET_PATH
    model_path = args[2] if len(args) > 2 else DEFAULT_MODEL_PATH
    if len(args) < 3:
        print(f'Usage: {args[0]} [dataset_path] [model_path]')
        print(f'Default dataset_path: {DEFAULT_TRAIN_DATASET_PATH}')
        print(f'Default model_path: {DEFAULT_MODEL_PATH}')

    main(dataset_path, model_path)

