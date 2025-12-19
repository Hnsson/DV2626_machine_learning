import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from src.data_loader import load_all_data
from src.utils import calculate_ranking_metrics, stratified_split, create_utility_matrix

from surprise import Dataset, Reader, KNNBasic, SVD
from collections import defaultdict

st.set_page_config(page_title="Movie Recommendation", layout="wide")

st.title("Movie Recommendation System")

# --- Data Loading ---
# This is done once at the top
ratings, users, movies = load_all_data()
NR_RATINGS_OUTLIER = 500

merged_df = ratings.merge(users, on="UserID", how="left").merge(
    movies, on="MovieID", how="left"
)
merged_df["UserRatingCount"] = merged_df.groupby("UserID")["Rating"].transform("count")
merged_df["Is_Outlier"] = (merged_df["AgeDesc"] == "Under 18") & (
    merged_df["UserRatingCount"] > NR_RATINGS_OUTLIER
)

train_df, test_df = stratified_split(merged_df, "UserID")


st.header("Select a Test User")

valid_user_ids = test_df["UserID"].unique()
valid_user_ids.sort()
user_meta = users.set_index("UserID")[["Gender", "Age", "OccupationDesc"]].to_dict(
    "index"
)


def user_format_func(user_id):
    if user_id in user_meta:
        u = user_meta[user_id]
        return f"({u['Gender']}) - \t{u['Age']} years old - {u['OccupationDesc']}"
    return f"ID {user_id}"


selected_user_id = st.selectbox(
    "Choose a User ID to analyze:",
    options=valid_user_ids,
    format_func=user_format_func,
    index=None,
    placeholder="Select a user to begin model training...",
)
if selected_user_id is None:
    st.info(
        "Please select a User ID above to train the models and generate predictions."
    )
    st.stop()
st.divider()

# --- Preprocessing Expander ---
with st.expander("Phase 4: Preprocessing & Splitting", expanded=False):
    st.header("1. Data Loading Summary")
    if ratings is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", int(users["UserID"].nunique()))
        col2.metric("Total Movies", int(movies["MovieID"].nunique()))
        col3.metric("Total Ratings", ratings.shape[0])
    else:
        st.error("Could not find data files.")

    st.header(f"2. Outlier Flagging (1 year old with >{NR_RATINGS_OUTLIER} ratings)")
    num_outliers = merged_df["Is_Outlier"].sum()
    st.markdown(f"Flagged **{num_outliers} ratings**")

    if num_outliers > 0:
        st.dataframe(
            merged_df[merged_df["Is_Outlier"] == True][
                ["UserID", "AgeDesc", "UserRatingCount", "Is_Outlier"]
            ].drop_duplicates(subset=["UserID"]),
            hide_index=True,
        )
    else:
        st.info("No outliers found with current threshold.")

    st.header("3. Train-Test Split")
    st.write(
        "Splitting the data into training and testing sets (80/20 split), stratified by user."
    )
    col1, col2 = st.columns(2)
    col1.caption(f"Train set shape: `{train_df.shape}`")
    col2.caption(f"Test set shape: `{test_df.shape}`")

    train_users = train_df["UserID"].nunique()
    test_users = test_df["UserID"].nunique()
    total_users = ratings["UserID"].nunique()

    col1.metric(
        label=f"Users in train set ({(train_users / total_users) * 100:.2f}%)",
        value=train_users,
    )
    col2.metric(
        label=f"Users in test set ({(test_users / total_users) * 100:.2f}%)",
        value=test_users,
    )

    st.header("3. Utility Matrix")
    st.write(
        "Creating the user-item utility matrix from the training set. Missing values are filled with 0."
    )
    utility_matrix = create_utility_matrix(train_df, users, movies)
    st.write("Utility matrix shape:", utility_matrix.shape)
    st.write("A peek at the utility matrix:")
    st.dataframe(utility_matrix.head())


# --- Baseline Models Expander ---
with st.expander("Phase 5: Baseline Models", expanded=False):
    st.write(
        """
        Before implementing complex models, it's crucial to establish a baseline.
        These simple models provide a benchmark to evaluate the performance of more sophisticated recommenders.
        """
    )

    # --- Global Mean Baseline ---
    st.subheader("Global Mean Baseline")
    mean_rating = train_df["Rating"].mean()
    test_df["MeanPrediction"] = mean_rating
    baseline_rmse = np.sqrt(
        mean_squared_error(test_df["Rating"], test_df["MeanPrediction"])
    )
    st.metric("Training Set Mean Rating", f"{mean_rating:.4f}")

    # --- Popularity Baseline ---
    st.subheader("Popularity Baseline (Non-Personalized)")
    st.info(
        """
        **Precision@K**: Out of the K movies we recommended, what fraction did the user *actually* like?
        I define "liked" as a rating of 4 or higher.
        """
    )

    @st.cache_data
    def get_top_k_popular(train_df, movies_df, k=10):
        movie_popularity = (
            train_df.groupby("MovieID").size().reset_index(name="RatingCount")
        )
        top_k = (
            movie_popularity.sort_values("RatingCount", ascending=False)
            .head(k)
            .merge(movies_df, on="MovieID")
        )
        return top_k

    def calculate_popularity_metrics(test_df, top_k_df, k=10, threshold=4.0):
        top_k_ids = set(top_k_df["MovieID"])
        user_groups = test_df.groupby("UserID")

        precisions = []
        recalls = []

        for _, group in user_groups:
            relevant_items = set(group[group["Rating"] >= threshold]["MovieID"])

            if not relevant_items:
                continue  # Skip users with no "likes" in test set (can't calculate recall)

            hits = len(top_k_ids.intersection(relevant_items))
            precisions.append(hits / k)
            recalls.append(hits / len(relevant_items))

        return np.mean(precisions), np.mean(recalls)

    top_10_popular = get_top_k_popular(train_df, movies, k=10)
    pop_prec, pop_rec = calculate_popularity_metrics(test_df, top_10_popular, 10, 4.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Global Mean RMSE (Lower is better)", f"{baseline_rmse:.4f}")
    c2.metric(
        "Top-10 Precision (Higher is better)",
        f"{pop_prec:.4f}",
        help="Baseline to beat",
    )
    c3.metric(
        "Top-10 Recall (Higher is better", f"{pop_rec:.4f}", help="Baseline to beat"
    )

    st.write("Top 10 Most Popular Movies:")
    st.dataframe(
        top_10_popular[["Title", "Genres", "RatingCount"]], use_container_width=True
    )

with st.expander("Phase 6: Collaborative Filtering (User-KNN)", expanded=False):
    st.caption("Evaluation of User-Based KNN vs Baseline")

    @st.cache_data
    def train_and_predict_knn(train_df, test_df):
        # 1. Re-create the Surprise objects inside (so Streamlit hashes the DataFrames correctly)
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(
            train_df[["UserID", "MovieID", "Rating"]], reader
        ).build_full_trainset()

        # 2. Train
        sim_options = {"name": "cosine", "user_based": True}
        algo_knn = KNNBasic(sim_options=sim_options, verbose=False)
        algo_knn.fit(train_data)

        # 3. Predict (Vectorized)
        test_set = list(zip(test_df["UserID"], test_df["MovieID"], test_df["Rating"]))
        predictions = algo_knn.test(test_set)

        return predictions

    with st.spinner("Training KNN & Generating Predictions..."):
        knn_predictions = train_and_predict_knn(train_df, test_df)
        test_df["KNN_Prediction"] = [pred.est for pred in knn_predictions]

    knn_rmse = np.sqrt(mean_squared_error(test_df["Rating"], test_df["KNN_Prediction"]))
    knn_prec, knn_rec = calculate_ranking_metrics(knn_predictions, 10, 4.0)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "User-KNN RMSE (Lower is better)",
        f"{knn_rmse:.4f}",
        delta=f"{mean_rating - knn_rmse:.4f} vs Baseline",
        delta_arrow="down",
    )
    delta_prec = (
        f"{knn_prec - pop_prec:.4f} vs Baseline" if "pop_prec" in locals() else None
    )
    c2.metric("Precision@10 (Higher is better)", f"{knn_prec:.4f}", delta=delta_prec)
    delta_rec = (
        f"{knn_rec - pop_rec:.4f} vs Baseline" if "pop_rec" in locals() else None
    )
    c3.metric("Recall@10 (Higher is better)", f"{knn_rec:.4f}", delta=delta_rec)

    example_user = selected_user_id
    st.write(f"**Example Prediction for User {example_user}:**")

    user_test_samples = test_df[test_df["UserID"] == example_user].head(5)
    st.dataframe(user_test_samples[["Title", "Rating", "KNN_Prediction"]])

with st.expander("Phase 7: Collaborative Filtering (Item-KNN)", expanded=False):
    st.caption("Evaluation of Item-Based KNN vs Baseline.")

    @st.cache_data
    def train_and_predict_item_knn(train_df, test_df):
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(
            train_df[["UserID", "MovieID", "Rating"]], reader
        ).build_full_trainset()

        sim_options = {"name": "cosine", "user_based": False}
        algo_item_knn = KNNBasic(sim_options=sim_options, verbose=False)
        algo_item_knn.fit(train_data)

        test_set = list(zip(test_df["UserID"], test_df["MovieID"], test_df["Rating"]))
        predictions = algo_item_knn.test(test_set)

        return predictions

    with st.spinner("Training Item-KNN & Generating Predictions..."):
        item_knn_predictions = train_and_predict_item_knn(train_df, test_df)
        test_df["Item_KNN_Prediction"] = [pred.est for pred in item_knn_predictions]

    item_knn_rmse = np.sqrt(
        mean_squared_error(test_df["Rating"], test_df["Item_KNN_Prediction"])
    )
    item_knn_prec, item_knn_rec = calculate_ranking_metrics(
        item_knn_predictions, 10, 4.0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Item-KNN RMSE (Lower is better)",
        f"{item_knn_rmse:.4f}",
        delta=f"{mean_rating - item_knn_rmse:.4f} vs Baseline",
        delta_arrow="down",
    )
    delta_prec = (
        f"{item_knn_prec - pop_prec:.4f} vs Baseline"
        if "pop_prec" in locals()
        else None
    )
    c2.metric(
        "Precision@10 (Higher is better)", f"{item_knn_prec:.4f}", delta=delta_prec
    )
    delta_rec = (
        f"{item_knn_rec - pop_rec:.4f} vs Baseline" if "pop_rec" in locals() else None
    )
    c3.metric("Recall@10 (Higher is better)", f"{item_knn_rec:.4f}", delta=delta_rec)
    st.caption("Comparison of Item-Based KNN vs Models.")

    if "knn_rmse" in locals():
        st.metric(
            label="Change vs User-KNN",
            value=f"{item_knn_rmse - knn_rmse:.4f}",  # The raw RMSE difference (e.g. -0.05)
            delta=f"{((item_knn_rmse - knn_rmse) / knn_rmse) * 100:.2f}%",  # The percentage change (e.g. -5.20%)
            delta_color="inverse",  # inverse: Negative (lower Error) = Green/Good
        )

    example_user = selected_user_id
    st.write(f"**Example Prediction for User {example_user}:**")

    user_test_samples = test_df[test_df["UserID"] == example_user].head(5)
    st.dataframe(
        user_test_samples[["Title", "Rating", "Item_KNN_Prediction", "KNN_Prediction"]]
    )


with st.expander("Phase 8: Matrix Factorization (SVD)", expanded=True):
    st.caption("Implementing SVD to discover hidden features")

    @st.cache_data
    def train_and_predict_svd(train_df, test_df):
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(
            train_df[["UserID", "MovieID", "Rating"]], reader
        ).build_full_trainset()

        algo_svd = SVD(n_factors=100, random_state=42)
        algo_svd.fit(train_data)

        test_set = list(zip(test_df["UserID"], test_df["MovieID"], test_df["Rating"]))
        predictions = algo_svd.test(test_set)

        return predictions

    with st.spinner("Training SVD & Generating Predictions..."):
        svd_predictions = train_and_predict_svd(train_df, test_df)
        test_df["SVD_Prediction"] = [pred.est for pred in svd_predictions]

    svd_rmse = np.sqrt(mean_squared_error(test_df["Rating"], test_df["SVD_Prediction"]))
    svd_prec, svd_rec = calculate_ranking_metrics(svd_predictions, 10, 4.0)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "SVD RMSE (Lower is better)",
        f"{svd_rmse:.4f}",
        delta=f"{mean_rating - svd_rmse:.4f} vs Baseline",
        delta_arrow="down",
    )
    delta_prec = (
        f"{svd_prec - pop_prec:.4f} vs Baseline" if "pop_prec" in locals() else None
    )
    c2.metric("Precision@10 (Higher is better)", f"{svd_prec:.4f}", delta=delta_prec)
    delta_rec = (
        f"{svd_rec - pop_rec:.4f} vs Baseline" if "pop_rec" in locals() else None
    )
    c3.metric("Recall@10 (Higher is better)", f"{svd_rec:.4f}", delta=delta_rec)

    st.caption("Comparison of SVD vs Models")
    col1, col2 = st.columns(2)
    if "knn_rmse" in locals():
        col1.metric(
            label="Change vs User-KNN",
            value=f"{svd_rmse - knn_rmse:.4f}",  # The raw RMSE difference (e.g. -0.05)
            delta=f"{((svd_rmse - knn_rmse) / knn_rmse) * 100:.2f}%",  # The percentage change (e.g. -5.20%)
            delta_color="inverse",  # inverse: Negative (lower Error) = Green/Good
        )
    if "item_knn_rmse" in locals():
        col2.metric(
            label="Change vs Item-KNN",
            value=f"{svd_rmse - item_knn_rmse:.4f}",  # The raw RMSE difference (e.g. -0.05)
            delta=f"{((svd_rmse - item_knn_rmse) / item_knn_rmse) * 100:.2f}%",  # The percentage change (e.g. -5.20%)
            delta_color="inverse",  # inverse: Negative (lower Error) = Green/Good
        )

    if "knn_rmse" in locals() and "item_knn_rmse" in locals():
        if svd_rmse < knn_rmse and svd_rmse < item_knn_rmse:
            st.success(
                "SVD outperformed Both Item- and User-KNN, validating the choice of Matrix Factorization for sparse data."
            )

    example_user = selected_user_id
    st.write(f"**Example Prediction for User {example_user}:**")

    user_test_samples = test_df[test_df["UserID"] == example_user].head(5)
    st.dataframe(
        user_test_samples[
            [
                "Title",
                "Rating",
                "SVD_Prediction",
                "Item_KNN_Prediction",
                "KNN_Prediction",
            ]
        ]
    )
