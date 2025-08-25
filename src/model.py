import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
import pandas as pd

def train_collaborative_filtering(ratings, n_factors=100, n_epochs=20, alpha=40):
    """
    Train collaborative filtering model using ALS from implicit
    with confidence weighting for explicit ratings
    """
    
    # Create mappers between IDs and matrix indices
    user_mapper = {u: i for i, u in enumerate(ratings['userId'].unique())}
    movie_mapper = {m: i for i, m in enumerate(ratings['movieId'].unique())}
    
    # Create inverse mappers
    user_inv_mapper = {i: u for u, i in user_mapper.items()}
    movie_inv_mapper = {i: m for m, i in movie_mapper.items()}
    
    # Create user-item matrix in COO format with confidence weighting
    row = [user_mapper[u] for u in ratings['userId']]
    col = [movie_mapper[m] for m in ratings['movieId']]
    
    # Convert ratings to confidence values (higher ratings = more confidence)
    confidence_data = 1 + alpha * ratings['rating'].values
    
    user_item_matrix = sp.coo_matrix(
        (confidence_data, (row, col)),
        shape=(len(user_mapper), len(movie_mapper))
    ).tocsr()
    
    # Initialize and train model
    model = AlternatingLeastSquares(
        factors=n_factors,
        iterations=n_epochs,
        random_state=42,
        # use_gpu=True,
        calculate_training_loss=True
    )
    model.fit(user_item_matrix)
    
    # Create mappers dictionary
    mappers = {
        'user_mapper': user_mapper,
        'movie_mapper': movie_mapper,
        'user_inv_mapper': user_inv_mapper,
        'movie_inv_mapper': movie_inv_mapper
    }
    
    return model, mappers, user_item_matrix

# # src/model.py
# import numpy as np
# import scipy.sparse as sp
# from implicit.als import AlternatingLeastSquares
# import pandas as pd

# def train_collaborative_filtering(ratings, n_factors=100, n_epochs=20, alpha=1.0):
#     """
#     Train collaborative filtering model using ALS from implicit
#     :param ratings: DataFrame with userId, movieId, rating
#     :param n_factors: Number of latent factors
#     :param n_epochs: Number of training epochs
#     :param alpha: Confidence scaling factor
#     :return: Tuple of (trained model, evaluation metrics, mappers)
#     """
#     # Create mappers between IDs and matrix indices
#     user_mapper = {u: i for i, u in enumerate(ratings['userId'].unique())}
#     movie_mapper = {m: i for i, m in enumerate(ratings['movieId'].unique())}
    
#     # Create inverse mappers
#     user_inv_mapper = {i: u for u, i in user_mapper.items()}
#     movie_inv_mapper = {i: m for m, i in movie_mapper.items()}
    
#     # Create user-item matrix in COO format
#     row = [user_mapper[u] for u in ratings['userId']]
#     col = [movie_mapper[m] for m in ratings['movieId']]
#     data = ratings['rating'].values
    
#     # Convert to confidence matrix (1 + alpha * rating)
#     confidence_data = 1.0 + alpha * data
#     user_item_matrix = sp.coo_matrix(
#         (confidence_data, (row, col)),
#         shape=(len(user_mapper), len(movie_mapper))
#     ).tocsr()
    
#     # Initialize and train model
#     model = AlternatingLeastSquares(
#         factors=n_factors,
#         iterations=n_epochs,
#         random_state=42
#     )
#     model.fit(user_item_matrix)
    
#     # Create mappers dictionary
#     mappers = {
#         'user_mapper': user_mapper,
#         'movie_mapper': movie_mapper,
#         'user_inv_mapper': user_inv_mapper,
#         'movie_inv_mapper': movie_inv_mapper
#     }
    
#     # Return model without metrics (implicit doesn't provide direct RMSE)
#     return model, mappers