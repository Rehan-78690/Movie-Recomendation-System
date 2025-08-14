# src/recommender.py
import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, cf_model, cf_mappers, cb_recommender, movies_df):
        """
        Initialize hybrid recommender
        :param cf_model: Trained ALS model
        :param cf_mappers: ID to index mappers
        :param cb_recommender: Content-based recommender
        :param movies_df: DataFrame of movie metadata
        """
        self.cf_model = cf_model
        self.cb_recommender = cb_recommender
        self.movies = movies_df
        self.cf_mappers = cf_mappers
        
    def recommend_for_user(self, user_id, top_n=10):
        """
        Generate hybrid recommendations for a user
        :param user_id: ID of target user
        :param top_n: Number of recommendations to return
        :return: DataFrame of recommended movies
        """
        # Get user index
        user_idx = self.cf_mappers['user_mapper'].get(user_id)
        if user_idx is None:
            return pd.DataFrame()  # New user - return empty
        
        # Generate CF scores
        user_factors = self.cf_model.user_factors[user_idx]
        scores = np.dot(self.cf_model.item_factors, user_factors)
        
        # Get all movie indices
        movie_indices = np.arange(scores.shape[0])
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'movie_index': movie_indices,
            'predicted_rating': scores
        })
        
        # Map indices to movie IDs
        predictions['movieId'] = predictions['movie_index'].map(
            self.cf_mappers['movie_inv_mapper'])
        
        # Filter out movies with no ID mapping
        predictions = predictions.dropna(subset=['movieId'])
        predictions['movieId'] = predictions['movieId'].astype(int)
        
        # Get top CF predictions
        cf_recs = predictions.sort_values('predicted_rating', ascending=False).head(top_n*3)
        
        # Blend with content-based similarity
        final_recs = []
        for _, row in cf_recs.iterrows():
            movie_id = row['movieId']
            similar_movies = self.cb_recommender.get_similar_movies(movie_id, top_n=1)
            
            if not similar_movies.empty:
                blended_score = (
                    0.7 * row['predicted_rating'] + 
                    0.3 * similar_movies['similarity_score'].values[0]
                )
                final_recs.append({
                    'movieId': movie_id,
                    'predicted_rating': row['predicted_rating'],
                    'blended_score': blended_score
                })
        
        # Sort by blended score and select top
        final_recs_df = pd.DataFrame(final_recs)
        final_recs_df = final_recs_df.sort_values('blended_score', ascending=False).head(top_n)
        
        # Merge movie details
        result = final_recs_df.merge(
            self.movies, 
            on='movieId'
        )[['movieId', 'title', 'genres', 'predicted_rating', 'blended_score']]
        
        return result
    
    def find_similar_movies(self, movie_title, top_n=10):
        """
        Find similar movies based on content
        :param movie_title: Title of reference movie
        :param top_n: Number of similar movies to return
        :return: DataFrame of similar movies
        """
        movie = self.movies[self.movies['title'].str.contains(movie_title, case=False)]
        if movie.empty:
            return pd.DataFrame()
        
        movie_id = movie.iloc[0]['movieId']
        return self.cb_recommender.get_similar_movies(movie_id, top_n)