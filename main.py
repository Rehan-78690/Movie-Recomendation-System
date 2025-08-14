# main.py
import pandas as pd
from src.data_loader import load_and_preprocess_data
from src.model import train_collaborative_filtering
from src.similarity import ContentBasedRecommender
from src.recommender import HybridRecommender

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    ratings, movies = load_and_preprocess_data(
        'data/ml-32m/ratings.csv', 
        'data/ml-32m/movies.csv',
        min_ratings=20,
        sample_size=0.1
    )
    
    # Train collaborative filtering model
    print("\nTraining collaborative filtering model...")
    cf_model, cf_mappers = train_collaborative_filtering(
        ratings,
        n_factors=100,
        n_epochs=20
    )
    print("Collaborative Filtering Model Trained")
    
    # Initialize content-based recommender
    print("\nBuilding content-based recommender...")
    cb_recommender = ContentBasedRecommender(movies)
    cb_recommender.build_tfidf_matrix()
    
    # Create hybrid recommender
    hybrid_rec = HybridRecommender(cf_model, cf_mappers, cb_recommender, movies)
    
    # Example recommendations
    user_id = 150
    movie_title = "The Dark Knight"
    
    print(f"\nHybrid recommendations for user {user_id}:")
    user_recs = hybrid_rec.recommend_for_user(user_id, top_n=5)
    print(user_recs[['title', 'genres', 'predicted_rating']])
    
    print(f"\nSimilar movies to '{movie_title}':")
    similar_movies = hybrid_rec.find_similar_movies(movie_title, top_n=5)
    print(similar_movies[['title', 'genres', 'similarity_score']])

if __name__ == "__main__":
    main()