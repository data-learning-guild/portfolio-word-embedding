"""Shared Functions and Classes"""
from pathlib import Path

import numpy as np


def mkdir_if_not_exist(path: str):
    """Make directory if not exist"""
    p = Path(path)
    if p.exists():
        return
    p.mkdir(parents=True, exist_ok=True)


def cos_similarity(_x: list, _y: list) -> float:
    """compute cos similarity"""
    vx = np.array(_x)
    vy = np.array(_y)
    return np.dot(vx, vy) / (np.linalg.norm(vx) * np.linalg.norm(vy))


def overlap(_x: list, _y: list) -> float:
    """overlap coefficient (Unuse)
        Szymkiewicz-Simpson coefficient)
        https://en.wikipedia.org/wiki/Overlap_coefficient
    """
    set_x = frozenset(_x)
    set_y = frozenset(_y)
    return len(set_x & set_y) / float(min(map(len, (set_x, set_y))))
