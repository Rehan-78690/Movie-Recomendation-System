import os
import pandas as pd
import numpy as np

def load_and_preprocess_data(ratings_path, movies_path, tags_path=None, min_ratings=10, sample_size=1.0):
    """
    Load and preprocess movie rating data with tags
    """
    # Load data
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Load tags if available
    if tags_path and os.path.exists(tags_path):
        tags = pd.read_csv(tags_path)
        # Aggregate tags by movie
        movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        movie_tags.columns = ['movieId', 'all_tags']
        movies = movies.merge(movie_tags, on='movieId', how='left')
        movies['all_tags'] = movies['all_tags'].fillna('')
    else:
        movies['all_tags'] = ''
    
    # Rest of your existing code remains the same...
    # Ensure correct data types
    ratings['movieId'] = ratings['movieId'].astype(int)
    movies['movieId'] = movies['movieId'].astype(int)
    
    # Sample data if needed
    if sample_size < 1.0:
        ratings = ratings.sample(frac=sample_size, random_state=42)
    
    # Filter movies with few ratings
    rating_counts = ratings['movieId'].value_counts()
    valid_movies = rating_counts[rating_counts >= min_ratings].index
    ratings = ratings[ratings['movieId'].isin(valid_movies)]
    
    # Calculate popularity and quality priors
    movie_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
    movie_stats.columns = ['n_ratings', 'avg_rating']
    
    # Calculate Bayesian average rating (IMDb style)
    C = ratings['rating'].mean()
    m = 50  # minimum number of ratings to consider
    movie_stats['bayes_rating'] = (
        (movie_stats['n_ratings'] / (movie_stats['n_ratings'] + m)) * movie_stats['avg_rating'] + 
        (m / (movie_stats['n_ratings'] + m)) * C
    )
    
    # Merge with movies data
    movies = movies.merge(movie_stats, on='movieId', how='left')
    
    # Fill NaN values for movies without enough ratings
    movies['n_ratings'] = movies['n_ratings'].fillna(0)
    movies['avg_rating'] = movies['avg_rating'].fillna(C)
    movies['bayes_rating'] = movies['bayes_rating'].fillna(C)
    
    # Parse genres into list
    movies['genres'] = movies['genres'].str.split('|')
    
    # Extract release year from title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
    movies['title_clean'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    
    return ratings, movies

# # src/data_loader.py
# import pandas as pd
# import numpy as np
# # from surprise import Dataset, Reader

# def load_and_preprocess_data(ratings_path, movies_path, min_ratings=10, sample_size=1.0):
#     """
#     Load and preprocess movie rating data
#     :param ratings_path: Path to ratings CSV
#     :param movies_path: Path to movies CSV
#     :param min_ratings: Minimum ratings a movie must have to be included
#     :param sample_size: Fraction of data to sample (for development)
#     :return: Tuple of (ratings_df, movies_df)
#     """
#     # Load data
#     ratings = pd.read_csv(ratings_path)
#     movies = pd.read_csv(movies_path)
#     ratings['movieId'] = ratings['movieId'].astype(int)  # Add this
#     movies['movieId'] = movies['movieId'].astype(int) 
#     # Sample data if needed
#     if sample_size < 1.0:
#         ratings = ratings.sample(frac=sample_size, random_state=42)
    
#     # Filter movies with few ratings
#     rating_counts = ratings['movieId'].value_counts()
#     valid_movies = rating_counts[rating_counts >= min_ratings].index
#     ratings = ratings[ratings['movieId'].isin(valid_movies)]
    
#     # Merge movie titles into ratings
#     ratings = ratings.merge(movies[['movieId', 'title']], on='movieId')
    
#     # Parse genres into list
#     movies['genres'] = movies['genres'].str.split('|')
    
#     # Extract release year from title
#     movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
#     movies['title_clean'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    
#     return ratings, movies