"""Build 128-dim feature vectors for each file node from its path tokens."""
import re
import hashlib
import numpy as np
from pathlib import PurePosixPath

TOKEN_RE = re.compile(r"[a-zA-Z]+")
FEAT_DIM = 128


def path_tokens(path):
    """e.g. 'pandas/core/dtypes/common.py' -> ['pandas','core','dtypes','common']"""
    parts = []
    for seg in PurePosixPath(path).parts:
        parts += TOKEN_RE.findall(seg.lower())
    return parts


def hash_token(tok, dim=FEAT_DIM):
    """Stable hash of token -> integer in [0, dim)."""
    h = hashlib.md5(tok.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % dim


def file_features(path, dim=FEAT_DIM):
    """Hash-based bag-of-path-tokens feature vector."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = path_tokens(path)
    if not tokens:
        return vec
    for tok in tokens:
        vec[hash_token(tok, dim)] += 1.0
    # L2 normalize
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec
