import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def evaluate_recommendations(recommender, test_ratings, k=10):
    """
    Evaluate recommendation quality using precision@k and recall@k
    """
    precisions = []
    recalls = []
    
    # For each user in test set
    for user_id in test_ratings['userId'].unique():
        user_ratings = test_ratings[test_ratings['userId'] == user_id]
        
        if len(user_ratings) < 2:  # Skip users with too few ratings
            continue
            
        # Consider ratings >= 4 as relevant
        relevant_items = set(user_ratings[user_ratings['rating'] >= 4]['movieId'].values)
        train_items = set(recommender.cf_mappers['movie_mapper'].keys())
        relevant_items = relevant_items.intersection(train_items)
        if not relevant_items:
            continue
        
        if not relevant_items:
            continue
            
        # Get recommendations
        recommendations = recommender.recommend_for_user(user_id, top_n=k)
        
        if recommendations.empty:
            continue
            
        recommended_items = set(recommendations['movieId'].values)
        
        
        # Calculate precision and recall
        relevant_recommended = recommended_items.intersection(relevant_items)
        precision = len(relevant_recommended) / k
        recall = len(relevant_recommended) / len(relevant_items)
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Return average precision and recall, avoiding division by zero
    if not precisions or not recalls:
        return 0.0, 0.0
    
    return np.mean(precisions), np.mean(recalls)