import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import linear_kernel


class ImprovedHybridRecommender:
    def __init__(self, cf_model, cf_mappers, cb_recommender, movies_df, ratings_df, user_item_matrix):
        self.cf_model = cf_model
        self.cb_recommender = cb_recommender
        self.movies = movies_df
        self.cf_mappers = cf_mappers
        self.user_item_matrix = user_item_matrix

        # Create mapping of seen items per user
        self.user2seen = ratings_df.groupby('userId')['movieId'].apply(set).to_dict()

        # Create cold start fallback (popular high-quality movies)
        self.popular_fallback = self.movies.sort_values(
            ['bayes_rating', 'n_ratings'], ascending=[False, False]
        ).head(100)

    def recommend_for_user(self, user_id, top_n=10):
        """Generate hybrid recommendations for a user"""
        # Get user index
        user_idx = self.cf_mappers['user_mapper'].get(user_id)
        if user_idx is None:
            return self._cold_start_recommendations(top_n)

        # Get seen items for this user
        seen = self.user2seen.get(user_id, set())

        # Get CF recommendations using implicit's built-in method
        try:
            cf_recommendations = self.cf_model.recommend(
                user_idx,
                self.user_item_matrix[user_idx],
                N=max(200, top_n * 20),  # larger pool for better hybrid/MMR
                filter_already_liked_items=True,
                recalculate_user=True
            )
        except Exception:
            return self._cold_start_recommendations(top_n)

        # --- Normalize cf_recommendations into two 1-D arrays: item_indices, scores ---
        if cf_recommendations is None:
            return self._cold_start_recommendations(top_n)

        if isinstance(cf_recommendations, tuple) and len(cf_recommendations) == 2:
            item_indices = np.asarray(cf_recommendations[0]).ravel()
            cf_scores = np.asarray(cf_recommendations[1]).ravel()
        elif hasattr(cf_recommendations, "shape"):
            arr = np.asarray(cf_recommendations)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                item_indices = arr[:, 0]
                cf_scores = arr[:, 1]
            else:
                return self._cold_start_recommendations(top_n)
        else:
            # Assume list-like of (id, score)
            try:
                item_indices = np.asarray([r[0] for r in cf_recommendations])
                cf_scores = np.asarray([r[1] for r in cf_recommendations])
            except Exception:
                return self._cold_start_recommendations(top_n)

        # If nothing came back, bail out to fallback
        if item_indices.size == 0:
            return self._cold_start_recommendations(top_n)

        # Map item *indices* -> movieId via movie_inv_mapper
        movie_inv = self.cf_mappers['movie_inv_mapper']
        cf_items = []
        cf_scores_filtered = []
        for idx, sc in zip(item_indices, cf_scores):
            idx = int(idx)
            if idx in movie_inv:  # safety
                cf_items.append(movie_inv[idx])
                cf_scores_filtered.append(float(sc))

        if not cf_items:
            return self._cold_start_recommendations(top_n)

        # Convert to DataFrame
        cf_df = pd.DataFrame({
            'movieId': cf_items,
            'cf_score': cf_scores_filtered
        })

        # Get content-based scores for these items
        cb_scores = self.cb_recommender.score_for_user(user_id, cf_items)

        # Merge CF and CB scores
        recommendations = cf_df.merge(cb_scores, on='movieId', how='left')
        recommendations = recommendations.merge(
            self.movies[['movieId', 'title', 'genres', 'n_ratings', 'bayes_rating']],
            on='movieId', how='left'
        )

        # Fill NaN values
        mean_bayes = float(self.movies['bayes_rating'].mean())
        recommendations['cf_score'] = recommendations['cf_score'].fillna(0)
        recommendations['cb_score'] = recommendations['cb_score'].fillna(0)
        recommendations['n_ratings'] = recommendations['n_ratings'].fillna(0)
        recommendations['bayes_rating'] = recommendations['bayes_rating'].fillna(mean_bayes)

        recommendations['cf_score_norm'] = minmax_scale(recommendations['cf_score'])
        recommendations['cb_score_norm'] = minmax_scale(recommendations['cb_score'])
        recommendations['pop_score_norm'] = minmax_scale(np.log1p(recommendations['n_ratings']))
        recommendations['quality_norm'] = minmax_scale(recommendations['bayes_rating'])

        recommendations['hybrid_score'] = (
            0.5 * recommendations['cf_score_norm'] +
            0.3 * recommendations['cb_score_norm'] +
            0.1 * recommendations['pop_score_norm'] +
            0.1 * recommendations['quality_norm']
        )

        # Apply MMR diversification
        final_recommendations = self._diversify_with_mmr(
            recommendations,
            top_n
        )

        return final_recommendations[['movieId', 'title', 'genres', 'hybrid_score']]

    def _cold_start_recommendations(self, top_n):
        """Fallback for new users - popular high-quality movies"""
        return self.popular_fallback[['movieId', 'title', 'genres', 'bayes_rating']].head(top_n).rename(
            columns={'bayes_rating': 'hybrid_score'}
        )

    def _diversify_with_mmr(self, recommendations, top_n, lambda_param=0.7):
        """Apply Maximal Marginal Relevance for diversification"""
        if len(recommendations) <= top_n:
            return recommendations.nlargest(top_n, 'hybrid_score')

        # Filter to movies that have TF-IDF vectors
        valid_movies = [mid for mid in recommendations['movieId']
                        if mid in self.cb_recommender.movie_index_map]
        recommendations = recommendations[recommendations['movieId'].isin(valid_movies)]

        if len(recommendations) <= top_n:
            return recommendations.nlargest(top_n, 'hybrid_score')

        # Get TF-IDF vectors for recommended movies
        movie_ids = recommendations['movieId'].tolist()
        movie_indices = [self.cb_recommender.movie_index_map[mid] for mid in movie_ids]
        movie_vectors = self.cb_recommender.tfidf_matrix[movie_indices]

        # Initialize
        selected = []
        remaining = list(range(len(movie_ids)))
        scores = recommendations['hybrid_score'].values

        # First select the highest scoring item
        first_idx = int(np.argmax(scores))
        selected.append(first_idx)
        if first_idx in remaining:
            remaining.remove(first_idx)

        # Select remaining items with MMR
        while len(selected) < top_n and remaining:
            mmr_scores = []

            for idx in remaining:
                # Relevance part
                relevance = scores[idx]

                # Diversity part (max similarity to already selected)
                if selected:
                    similarities = linear_kernel(
                        movie_vectors[idx:idx + 1],
                        movie_vectors[selected]
                    ).flatten()
                    diversity = float(np.max(similarities))
                else:
                    diversity = 0.0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append(mmr_score)

            # Select item with highest MMR score
            best_remaining_idx = remaining[int(np.argmax(mmr_scores))]
            selected.append(best_remaining_idx)
            remaining.remove(best_remaining_idx)

        return recommendations.iloc[selected]

    def find_similar_movies(self, movie_title, top_n=10):
        """Find similar movies with a balanced approach (series boost + content similarity)."""
        movie = self.movies[self.movies['title'].str.contains(movie_title, case=False, na=False)]
        if movie.empty:
            return pd.DataFrame()

        movie = movie.sort_values(['n_ratings', 'bayes_rating'], ascending=[False, False])
        movie_id = int(movie.iloc[0]['movieId'])

        # First try to find series movies
        series_movies = self.cb_recommender.find_series_movies(movie.iloc[0]['title'], top_n=top_n)
        if series_movies is None:
            series_movies = pd.DataFrame()

        # Get regular content-based similar movies (get more for blending)
        content_movies = self.cb_recommender.get_similar_movies(movie_id, top_n * 2)
        if content_movies is None:
            content_movies = pd.DataFrame()

        # If one side is empty, return the other
        if series_movies.empty and content_movies.empty:
            return pd.DataFrame()
        if series_movies.empty:
            out = content_movies.head(top_n).copy()
            out = out.rename(columns={'similarity_score': 'similarity_score'})
            return out[['movieId', 'title', 'genres', 'similarity_score']]
        if content_movies.empty:
            out = series_movies.head(top_n).copy()
            # series finder returns 'combined_score'; align to 'similarity_score'
            if 'combined_score' in out.columns:
                out = out.rename(columns={'combined_score': 'similarity_score'})
            else:
                out['similarity_score'] = 1.0
            return out[['movieId', 'title', 'genres', 'similarity_score']]

        # Prepare frames
        series_movies = series_movies.copy()
        content_movies = content_movies.copy()

        series_movies['is_series'] = True
        content_movies['is_series'] = False

        # Normalize scores for fair comparison
        if 'similarity_score' in content_movies.columns and content_movies['similarity_score'].notna().any():
            content_movies['normalized_score'] = minmax_scale(content_movies['similarity_score'].fillna(0.0))
        else:
            content_movies['normalized_score'] = 0.5  # neutral fallback

        if 'combined_score' in series_movies.columns and series_movies['combined_score'].notna().any():
            series_movies['normalized_score'] = minmax_scale(series_movies['combined_score'].fillna(0.0))
        elif 'similarity_score' in series_movies.columns and series_movies['similarity_score'].notna().any():
            series_movies['normalized_score'] = minmax_scale(series_movies['similarity_score'].fillna(0.0))
        else:
            series_movies['normalized_score'] = 0.8  # small bias toward series when unknown

        # Give series movies a slight boost
        series_movies['normalized_score'] = series_movies['normalized_score'] * 1.2

        # Combine and deduplicate
        combined = pd.concat([series_movies, content_movies], ignore_index=True, sort=False)
        combined = combined.drop_duplicates('movieId', keep='first')

        # Sort by normalized score and return top results
        combined = combined.sort_values('normalized_score', ascending=False).head(top_n)

        # Align output to have 'similarity_score'
        combined = combined.rename(columns={'normalized_score': 'similarity_score'})
        return combined[['movieId', 'title', 'genres', 'similarity_score']]
