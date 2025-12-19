from collections import defaultdict
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


def calculate_ranking_metrics(predictions, k=10, threshold=4.0):
    # Map the predictions to each user
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return np.mean(list(precisions.values())), np.mean(list(recalls.values()))
