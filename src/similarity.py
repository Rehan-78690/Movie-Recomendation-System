from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
import pandas as pd
import re
import numpy as np
from scipy.sparse import csr_matrix


class EnhancedContentRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.movie_index_map = None
        self.user_profiles = None

    def build_tfidf_matrix(self):
        """Create TF-IDF matrix from movie genres and titles"""
        # Create text representation: title + genres
        self.movies = self.movies.copy()
        self.movies['content'] = (
            self.movies['title_clean'].fillna('') + ' ' +
            self.movies['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        )

        # Initialize and fit TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_features=3000
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['content'])

        # Create movie index mapping
        self.movie_index_map = pd.Series(
            self.movies.index,
            index=self.movies['movieId']
        ).to_dict()

    def build_user_profile(self, ratings_df, min_rating=4.0):
        """Build user profiles based on their rated movies"""
        self.user_profiles = {}

        # Filter for highly rated movies
        liked_ratings = ratings_df[ratings_df['rating'] >= min_rating]

        # Group by user
        for user_id, user_ratings in liked_ratings.groupby('userId'):
            movie_ids = user_ratings['movieId'].values

            # Get indices of movies in our matrix
            movie_indices = [
                self.movie_index_map.get(mid) for mid in movie_ids
                if mid in self.movie_index_map
            ]
            movie_indices = [idx for idx in movie_indices if idx is not None]

            if not movie_indices:
                continue

            # Create user profile as average of movie vectors
            user_vector = self.tfidf_matrix[movie_indices].mean(axis=0)

            # Convert to array and normalize
            user_vector_array = np.asarray(user_vector).flatten()
            user_vector_normalized = normalize(user_vector_array.reshape(1, -1))

            self.user_profiles[user_id] = user_vector_normalized

    def get_similar_movies(self, movie_id, top_n=10):
        """Find similar movies using cosine similarity"""
        if movie_id not in self.movie_index_map:
            return pd.DataFrame()

        idx = self.movie_index_map[movie_id]

        # Calculate cosine similarities
        cosine_similarities = linear_kernel(
            self.tfidf_matrix[idx:idx + 1],
            self.tfidf_matrix
        ).flatten()

        # Get top similar movies (exclude itself at position 0)
        similar_indices = cosine_similarities.argsort()[::-1][1:top_n + 1]
        similar_movies = self.movies.iloc[similar_indices].copy()
        similar_movies['similarity_score'] = cosine_similarities[similar_indices]

        return similar_movies[['movieId', 'title', 'genres', 'similarity_score']]

    def score_for_user(self, user_id, candidate_movie_ids=None):
        """Calculate content-based scores for a user against movies"""
        if user_id not in self.user_profiles:
            return pd.DataFrame(columns=['movieId', 'cb_score'])

        user_vector = self.user_profiles[user_id]

        if candidate_movie_ids is not None:
            # Filter candidate IDs that exist in our index map
            kept_ids = [mid for mid in candidate_movie_ids if mid in self.movie_index_map]
            if not kept_ids:
                return pd.DataFrame(columns=['movieId', 'cb_score'])

            # Get indices for kept IDs
            idxs = [self.movie_index_map[mid] for mid in kept_ids]
            cand_mat = self.tfidf_matrix[idxs]

            # Calculate scores
            scores = linear_kernel(user_vector, cand_mat).flatten()

            return pd.DataFrame({'movieId': kept_ids, 'cb_score': scores})
        else:
            # Score all movies
            scores = linear_kernel(user_vector, self.tfidf_matrix).flatten()
            return pd.DataFrame({
                'movieId': self.movies['movieId'],
                'cb_score': scores
            })

    def find_series_movies(self, movie_title, top_n=5):
        """
        Heuristic finder for sequels/entries in the same series/franchise.
        Combines title-pattern matches + genre Jaccard + simple title-word Jaccard.
        """
        source_movie = self.movies[self.movies['title'] == movie_title]
        if source_movie.empty:
            return pd.DataFrame()

        source_movie = source_movie.iloc[0]
        source_movie_id = int(source_movie['movieId'])
        source_genres = source_movie['genres']
        if not isinstance(source_genres, list):
            source_genres = str(source_genres).split('|')
        source_genres = set([g for g in source_genres if g])

        # Clean base title (remove year and leading articles)
        base_title = re.sub(r'\(.*?\)', '', movie_title).strip()
        base_title = re.sub(r'^(The|A|An)\s+', '', base_title, flags=re.IGNORECASE).strip()

        patterns = [
            rf"{re.escape(base_title)}.*\b(Part|II|III|IV|V|VI|2|3|4|5|6|:\s)",
            rf"{re.escape(base_title)}.*\d{4}.*\d{4}",
            rf"\b(Part|II|III|IV|V|VI|2|3|4|5|6|:\s).*{re.escape(base_title)}"
        ]

        potential_series = pd.DataFrame()
        for pattern in patterns:
            try:
                matches = self.movies[self.movies['title'].str.contains(pattern, case=False, regex=True, na=False)]
                if not matches.empty:
                    potential_series = pd.concat([potential_series, matches], ignore_index=True)
            except re.error:
                # Skip invalid regex edge cases
                continue

        # Drop duplicates, exclude the source title itself
        if potential_series.empty:
            return pd.DataFrame()

        potential_series = potential_series.drop_duplicates(subset=['movieId'])
        potential_series = potential_series[potential_series['title'] != movie_title]

        if potential_series.empty:
            return pd.DataFrame()

        # --- Genre similarity (Jaccard) ---
        def calculate_genre_similarity(target_genres):
            tg = target_genres if isinstance(target_genres, list) else str(target_genres).split('|')
            tg_set = set([g for g in tg if g])
            intersection = len(source_genres.intersection(tg_set))
            union = len(source_genres.union(tg_set))
            return intersection / union if union > 0 else 0.0

        potential_series['genre_similarity'] = potential_series['genres'].apply(calculate_genre_similarity)
        potential_series = potential_series[potential_series['genre_similarity'] >= 0.4]
        if potential_series.empty:
            return pd.DataFrame()

        # --- Title similarity (word-level Jaccard) ---
        def calculate_title_similarity(target_title):
            target_clean = re.sub(r'\(.*?\)', '', str(target_title)).strip()
            target_clean = re.sub(r'^(The|A|An)\s+', '', target_clean, flags=re.IGNORECASE).strip()
            source_words = set(base_title.lower().split())
            target_words = set(target_clean.lower().split())
            intersection = len(source_words.intersection(target_words))
            union = len(source_words.union(target_words))
            return intersection / union if union > 0 else 0.0

        potential_series['title_similarity'] = potential_series['title'].apply(calculate_title_similarity)

        # Combined score: 60% title similarity + 40% genre similarity
        potential_series['combined_score'] = (
            0.5 * potential_series['title_similarity'] +
            0.3 * potential_series['genre_similarity']
        )

        # Sort by combined score and get top results
        potential_series = potential_series.sort_values('combined_score', ascending=False)

        return potential_series[['movieId', 'title', 'genres', 'combined_score']].head(top_n)
