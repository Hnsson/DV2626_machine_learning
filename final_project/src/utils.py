import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


@st.cache_data
def stratified_split(df, group, frac=0.8):
    """Splits a dataframe by a group with a given fraction."""
    train = df.groupby(group).sample(frac=frac, random_state=42)
    test = df.drop(train.index)
    return train, test


@st.cache_data
def create_utility_matrix(data_split, users, movies):
    utility_matrix = data_split.pivot_table(
        index="UserID", columns="MovieID", values="Rating"
    )

    user_ids = users["UserID"].unique()
    utility_matrix = utility_matrix.reindex(index=user_ids, fill_value=0)

    movie_ids = movies["MovieID"].unique()
    utility_matrix = utility_matrix.reindex(columns=movie_ids, fill_value=0)

    return utility_matrix.fillna(0)


def evaluate_model_predictions(test_df, predictions_dict, k=10):
    # Calculate Precision@K (Ranking Metric)
    # Get the set of movies each user ACTUALLY liked (Rating >= 4) in Test
    user_liked_actual = (
        test_df[test_df["Rating"] >= 4]
        .groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    precisions = []
    for user_id, recommended_items in predictions_dict.items():
        # Get what this user liked in the test set
        actual_likes = user_liked_actual.get(user_id, set())

        if not actual_likes:
            continue  # Skip users who didn't like anything in test (can't evaluate)

        # Count hits
        hits = len(actual_likes.intersection(set(recommended_items)))
        precisions.append(hits / k)

    precision_score = np.mean(precisions) if precisions else 0
    return precision_score
