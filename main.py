import os
# Use 1 thread as implicit warns about BLAS threadpools
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

from src.cache_utils import (
    load_or_preprocess,
    load_or_train_als,
    load_or_build_tfidf,
    build_user_profiles_cached,
)
from src.recommender import ImprovedHybridRecommender
from src.evaluation import evaluate_recommendations


def main():
    force_refresh = os.getenv("RECACHE", "0") == "1"

    # 1) Data (cached)
    print("Loading and preprocessing data (cached)...")
    ratings, movies, data_key = load_or_preprocess(
        'data/ml-32m/ratings.csv',
        'data/ml-32m/movies.csv',
        min_ratings=30,
        sample_size=0.5,
        force_refresh=force_refresh,
    )

    # 2) Train/test split (fast; do each run)
    from numpy.random import default_rng
    def per_user_split(ratings_df, test_ratio=0.2, min_interactions=5, seed=42):
        rng = default_rng(seed)
        train_indices, test_indices = [], []
        for uid, group in ratings_df.groupby('userId'):
            if len(group) < min_interactions:
                train_indices.extend(group.index)
            else:
                test_size = max(1, int(len(group) * test_ratio))
                test_idx = rng.choice(group.index, size=test_size, replace=False)
                train_idx = list(set(group.index) - set(test_idx))
                train_indices.extend(train_idx)
                test_indices.extend(test_idx)
        return ratings_df.loc[train_indices], ratings_df.loc[test_indices]

    train_ratings, test_ratings = per_user_split(ratings, test_ratio=0.2)

    # 3) ALS (cached)
    print("\nTraining/loading ALS (cached)...")
    cf_model, cf_mappers, user_item_matrix = load_or_train_als(
        train_ratings,
        data_key=data_key,
        factors=100,
        epochs=20,
        alpha=40,
        force_refresh=force_refresh,
    )
    print("ALS ready.")

    # 4) TF-IDF (cached) + user profiles (optionally cached)
    print("\nBuilding/loading TF-IDF (cached)...")
    cb_recommender = load_or_build_tfidf(
        movies,
        data_key=data_key,
        code_version="tfidf_v1",  # bump if you change vectorizer params/weights
        force_refresh=force_refresh,
    )
    print("TF-IDF ready.")

    print("\nBuilding user profiles (cached)...")
    build_user_profiles_cached(
        cb_recommender,
        train_ratings,
        min_rating=4.0,
        data_key=data_key,
        code_version="uprofile_v1",
        force_refresh=force_refresh,
    )
    print("User profiles ready.")

    # 5) Hybrid recommender
    hybrid_rec = ImprovedHybridRecommender(
        cf_model,
        cf_mappers,
        cb_recommender,
        movies,
        train_ratings,
        user_item_matrix
    )

    # Example recommendations
    user_id = 150
    movie_title = "Batman"

    print(f"\nHybrid recommendations for user {user_id}:")
    user_recs = hybrid_rec.recommend_for_user(user_id, top_n=5)
    print(user_recs[['title', 'genres', 'hybrid_score']])

    print(f"\nSimilar movies to '{movie_title}':")
    similar_movies = hybrid_rec.find_similar_movies(movie_title, top_n=5)
    if not similar_movies.empty:
        if 'similarity_score' in similar_movies.columns:
            print(similar_movies[['title', 'genres', 'similarity_score']])
        else:
            print(similar_movies[['title', 'genres']])
    else:
        print("No similar movies found.")

    # Evaluate
    print("\nEvaluating recommendation quality...")
    precision, recall = evaluate_recommendations(hybrid_rec, test_ratings)
    print(f"Precision@10: {precision:.4f}, Recall@10: {recall:.4f}")


if __name__ == "__main__":
    main()
