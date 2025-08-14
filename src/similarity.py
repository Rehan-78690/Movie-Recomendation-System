# src/similarity.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

class ContentBasedRecommender:
    def __init__(self, movies_df):
        """
        Initialize content-based recommender
        :param movies_df: DataFrame containing movie metadata
        """
        self.movies = movies_df
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.movie_index_map = None
    
    def build_tfidf_matrix(self):
        """Create TF-IDF matrix from movie genres and titles"""
        # Create text representation: title + genres
        self.movies['content'] = self.movies['title_clean'] + ' ' + \
                                self.movies['genres'].apply(lambda x: ' '.join(x))
        
        # Initialize and fit TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['content'])
        
        # Create movie index mapping
        self.movie_index_map = pd.Series(
            self.movies.index, 
            index=self.movies['movieId']
        ).to_dict()
    
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Find similar movies using cosine similarity
        :param movie_id: ID of reference movie
        :param top_n: Number of similar movies to return
        :return: DataFrame of similar movies
        """
        if movie_id not in self.movie_index_map:
            return pd.DataFrame()
        
        idx = self.movie_index_map[movie_id]
        
        # Calculate cosine similarities
        cosine_similarities = linear_kernel(
            self.tfidf_matrix[idx:idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar movies
        similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]
        similar_movies = self.movies.iloc[similar_indices].copy()
        similar_movies['similarity_score'] = cosine_similarities[similar_indices]
        
        return similar_movies[['movieId', 'title', 'genres', 'similarity_score']]