# src/data_loader.py
import pandas as pd
import numpy as np
# from surprise import Dataset, Reader

def load_and_preprocess_data(ratings_path, movies_path, min_ratings=10, sample_size=1.0):
    """
    Load and preprocess movie rating data
    :param ratings_path: Path to ratings CSV
    :param movies_path: Path to movies CSV
    :param min_ratings: Minimum ratings a movie must have to be included
    :param sample_size: Fraction of data to sample (for development)
    :return: Tuple of (ratings_df, movies_df)
    """
    # Load data
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    ratings['movieId'] = ratings['movieId'].astype(int)  # Add this
    movies['movieId'] = movies['movieId'].astype(int) 
    # Sample data if needed
    if sample_size < 1.0:
        ratings = ratings.sample(frac=sample_size, random_state=42)
    
    # Filter movies with few ratings
    rating_counts = ratings['movieId'].value_counts()
    valid_movies = rating_counts[rating_counts >= min_ratings].index
    ratings = ratings[ratings['movieId'].isin(valid_movies)]
    
    # Merge movie titles into ratings
    ratings = ratings.merge(movies[['movieId', 'title']], on='movieId')
    
    # Parse genres into list
    movies['genres'] = movies['genres'].str.split('|')
    
    # Extract release year from title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
    movies['title_clean'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    
    return ratings, movies