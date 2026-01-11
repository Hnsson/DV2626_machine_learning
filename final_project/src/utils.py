from collections import defaultdict
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


@st.cache_data
def stratified_split(df, group, frac=0.8):
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


@st.cache_resource
def train_user_clustering(users_df, n_clusters=5):
    df = users_df.copy()
    df["Gender_Code"] = df["Gender"].map({"F": 0, "M": 1})

    scaler = StandardScaler()
    features = df[["Gender_Code", "Age", "Occupation"]]
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_features)

    return kmeans, scaler, df


@st.cache_data
def get_cluster_top_movies(cluster_id, clustered_users_df, train_df, movies_df, k=10):
    cluster_users = clustered_users_df[clustered_users_df["Cluster"] == cluster_id][
        "UserID"
    ]
    cluster_ratings = train_df[train_df["UserID"].isin(cluster_users)]

    movie_stats = (
        cluster_ratings.groupby("MovieID")
        .agg(RatingCount=("Rating", "count"), MeanRating=("Rating", "mean"))
        .reset_index()
    )

    popular_in_cluster = movie_stats[movie_stats["RatingCount"] > 5]

    top_movies = popular_in_cluster.sort_values(
        ["MeanRating", "RatingCount"], ascending=[False, False]
    ).head(k)

    return top_movies.merge(movies_df, on="MovieID")


def get_clustering_predictions(train_df, test_df, clustered_users):
    train_w_cluster = train_df.merge(
        clustered_users[["UserID", "Cluster"]], on="UserID"
    )

    cluster_movie_means = (
        train_w_cluster.groupby(["Cluster", "MovieID"])["Rating"].mean().reset_index()
    )
    cluster_movie_means.rename(columns={"Rating": "EstRating"}, inplace=True)

    test_w_cluster = test_df.merge(clustered_users[["UserID", "Cluster"]], on="UserID")
    predictions_df = test_w_cluster.merge(
        cluster_movie_means, on=["Cluster", "MovieID"], how="left"
    )

    global_mean = train_df["Rating"].mean()
    predictions_df["EstRating"] = predictions_df["EstRating"].fillna(global_mean)

    formatted_preds = []
    for row in predictions_df.itertuples():
        formatted_preds.append(
            (row.UserID, row.MovieID, row.Rating, row.EstRating, None)
        )
    return formatted_preds


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
