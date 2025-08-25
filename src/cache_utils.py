import os
import json
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import pandas as pd
import numpy as np
from scipy import sparse

from src.data_loader import load_and_preprocess_data
from src.model import train_collaborative_filtering
from src.similarity import EnhancedContentRecommender


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_dict(d: Dict[str, Any]) -> str:
    """Stable short hash for dicts (order-independent)."""
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return -1.0


def _dump_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _dump_joblib(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def _load_joblib(path: Path):
    return joblib.load(path)


def _dump_npz_csr(matrix: sparse.csr_matrix, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, matrix)


def _load_npz_csr(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(path)


def load_or_preprocess(
    ratings_path: str,
    movies_path: str,
    min_ratings: int = 30,
    sample_size: float = 1.0,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Cache layer for load_and_preprocess_data(). Returns (ratings, movies, data_key).
    Invalidates on file mtime or param changes.
    """
    meta = {
        "ratings_path": ratings_path,
        "movies_path": movies_path,
        "ratings_mtime": _file_mtime(ratings_path),
        "movies_mtime": _file_mtime(movies_path),
        "min_ratings": min_ratings,
        "sample_size": sample_size,
        "version": "pre_v1",  # bump if you change preprocessing logic
    }
    key = _hash_dict(meta)

    r_path = CACHE_DIR / f"ratings_{key}.parquet"
    m_path = CACHE_DIR / f"movies_{key}.parquet"
    meta_path = CACHE_DIR / f"data_meta_{key}.json"

    if not force_refresh and r_path.exists() and m_path.exists():
        try:
            ratings = _load_parquet(r_path)
            movies = _load_parquet(m_path)
            return ratings, movies, key
        except Exception:
            pass  # fall through to rebuild

    ratings, movies = load_and_preprocess_data(
        ratings_path, movies_path, min_ratings=min_ratings, sample_size=sample_size
    )
    _dump_parquet(ratings, r_path)
    _dump_parquet(movies, m_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return ratings, movies, key


def load_or_train_als(
    train_ratings: pd.DataFrame,
    data_key: str,
    factors: int = 100,
    epochs: int = 20,
    alpha: int = 40,
    force_refresh: bool = False,
):
    """
    Cache layer for ALS training: returns (model, mappers, user_item_matrix).
    Keyed by data_key + params.
    """
    meta = {
        "data_key": data_key,
        "factors": factors,
        "epochs": epochs,
        "alpha": alpha,
        "version": "als_v1",  # bump if you change training logic
        # A lightweight fingerprint of the training set:
        "n_rows": int(len(train_ratings)),
        "n_users": int(train_ratings["userId"].nunique()),
        "n_items": int(train_ratings["movieId"].nunique()),
    }
    key = _hash_dict(meta)

    model_path = CACHE_DIR / f"als_model_{key}.joblib"
    mappers_path = CACHE_DIR / f"als_mappers_{key}.joblib"
    uim_path = CACHE_DIR / f"user_item_{key}.npz"
    meta_path = CACHE_DIR / f"als_meta_{key}.json"

    if not force_refresh and model_path.exists() and mappers_path.exists() and uim_path.exists():
        try:
            model = _load_joblib(model_path)
            mappers = _load_joblib(mappers_path)
            user_item = _load_npz_csr(uim_path)
            return model, mappers, user_item
        except Exception:
            pass

    model, mappers, user_item = train_collaborative_filtering(
        train_ratings, n_factors=factors, n_epochs=epochs, alpha=alpha
    )

    _dump_joblib(model, model_path)
    _dump_joblib(mappers, mappers_path)
    _dump_npz_csr(user_item, uim_path)
    meta_path.write_text(json.dumps(meta, indent=2))

    return model, mappers, user_item


def load_or_build_tfidf(
    movies: pd.DataFrame,
    data_key: str,
    code_version: str = "tfidf_v1",
    force_refresh: bool = False,
) -> EnhancedContentRecommender:
    """
    Cache the fitted EnhancedContentRecommender (matrix + vectorizers + maps).
    """
    meta = {
        "data_key": data_key,
        "code_version": code_version,  # bump if you change vectorizer params/weights
        "n_movies": int(len(movies)),
    }
    key = _hash_dict(meta)
    rec_path = CACHE_DIR / f"cb_recommender_{key}.joblib"
    meta_path = CACHE_DIR / f"tfidf_meta_{key}.json"

    if not force_refresh and rec_path.exists():
        try:
            return _load_joblib(rec_path)
        except Exception:
            pass

    cb = EnhancedContentRecommender(movies.copy())
    cb.build_tfidf_matrix()
    _dump_joblib(cb, rec_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return cb


def build_user_profiles_cached(
    cb_recommender: EnhancedContentRecommender,
    train_ratings: pd.DataFrame,
    min_rating: float = 4.0,
    data_key: str = "",
    code_version: str = "uprofile_v1",
    force_refresh: bool = False,
):
    """
    Optionally cache user_profiles (dict: userId -> vector). This is already fast,
    but caching helps on very large datasets.
    """
    # Create a compact hash of the ratings triplets (userId, movieId, rating)
    # Enough to detect changes without storing all data.
    tiny_df = train_ratings[['userId', 'movieId', 'rating']].copy()
    tiny_df = tiny_df.sort_values(['userId', 'movieId']).reset_index(drop=True)
    fp = hashlib.sha1(pd.util.hash_pandas_object(tiny_df, index=False).values.tobytes()).hexdigest()[:12]

    meta = {
        "data_key": data_key,
        "code_version": code_version,
        "min_rating": float(min_rating),
        "fp": fp,
    }
    key = _hash_dict(meta)
    path = CACHE_DIR / f"user_profiles_{key}.joblib"
    meta_path = CACHE_DIR / f"user_profiles_meta_{key}.json"

    if not force_refresh and path.exists():
        try:
            cb_recommender.user_profiles = _load_joblib(path)
            return
        except Exception:
            pass

    cb_recommender.build_user_profile(train_ratings, min_rating=min_rating)
    _dump_joblib(cb_recommender.user_profiles, path)
    meta_path.write_text(json.dumps(meta, indent=2))
