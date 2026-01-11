import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from src.data_loader import load_all_data
from src.utils import (
    calculate_ranking_metrics,
    stratified_split,
    create_utility_matrix,
    train_user_clustering,
    get_cluster_top_movies,
    get_clustering_predictions,
)

from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import GridSearchCV

st.set_page_config(page_title="Movie Recommendation", layout="wide")

st.title("Movie Recommendation System")

# --- Data Loading ---
ratings, users, movies = load_all_data()
NR_RATINGS_OUTLIER = 500

merged_df = ratings.merge(users, on="UserID", how="left").merge(
    movies, on="MovieID", how="left"
)
merged_df["UserRatingCount"] = merged_df.groupby("UserID")["Rating"].transform("count")

# Flagging outliers, (Under 18 and alot of ratings)
merged_df["Is_Outlier"] = (merged_df["AgeDesc"] == "Under 18") & (
    merged_df["UserRatingCount"] > NR_RATINGS_OUTLIER
)

# The stratified splitting by UserID
train_df, test_df = stratified_split(merged_df, "UserID")


selected_user_id = 6028  # 18 year old college student (just for test)

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

    st.header(
        f"2. Outlier Flagging (Under 18 years old with >{NR_RATINGS_OUTLIER} ratings)"
    )
    num_outliers = merged_df["Is_Outlier"].sum()

    if num_outliers > 0:
        st.dataframe(
            merged_df[merged_df["Is_Outlier"] == True][
                ["UserID", "AgeDesc", "Age", "UserRatingCount", "Is_Outlier"]
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
    st.dataframe(utility_matrix.head())
with st.expander(
    "Phase 4b: Model Configuration & Hyperparameter Tuning", expanded=False
):
    st.caption(
        """
        To ensure optimal performance, I used a **Grid Search Cross-Validation** to select the best hyperparameters for the algorithms.
        """
    )

    @st.cache_data
    def run_tuning(df, algorithm_name):
        # Grid Search (on sample of 20k because 1M ratings took wayyyyy too long)
        subset_df = df.sample(n=20000, random_state=42)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(subset_df[["UserID", "MovieID", "Rating"]], reader)

        if algorithm_name == "SVD":
            param_grid = {
                "n_factors": [20, 50, 100],  # Latent Factors
                "lr_all": [0.005, 0.01],  # Learning Rate
            }
            algo_class = SVD

        elif algorithm_name == "KNN":
            param_grid = {
                "k": [20, 40, 60],
                "sim_options": {
                    "name": ["cosine", "pearson"],
                    "user_based": [True, False],  # Comparing user-based variants
                },
            }
            algo_class = KNNBasic

        gs = GridSearchCV(algo_class, param_grid, measures=["rmse"], cv=3)
        gs.fit(data)

        return (
            gs.best_score["rmse"],
            gs.best_params["rmse"],
            pd.DataFrame.from_dict(gs.cv_results),
        )

    tab1, tab2 = st.tabs(["SVD Tuning", "KNN Tuning"])

    # --- SVD TUNING ---
    with tab1:
        st.write(
            "Optimize **Latent Factors**, **Learning Rate**, and **Regularization**."
        )

        if st.button("Run SVD Grid Search"):
            with st.spinner("Running 3-Fold Cross-Validation on SVD..."):
                best_score, best_params, results_df = run_tuning(merged_df, "SVD")

            st.success(f"**Best RMSE found:** {best_score:.6f}")
            st.json(best_params)

            st.write("Grid Search Results (Top 5 Configurations):")
            st.dataframe(
                results_df[
                    [
                        "param_n_factors",
                        "param_lr_all",
                        "mean_test_rmse",
                        "mean_fit_time",
                    ]
                ]
                .sort_values("mean_test_rmse")
                .head(5)
                .style.highlight_min(subset=["mean_test_rmse"], color="#378353")
                .format({"param_lr_all": "{:.2f}"}),
                width="stretch",
            )

    # --- KNN TUNING ---
    with tab2:
        st.write("Optimize **k (Neighbors)** and **Similarity Metric**.")

        if st.button("Run KNN Grid Search"):
            with st.spinner("Running 3-Fold Cross-Validation on KNN..."):
                best_score, best_params, results_df = run_tuning(merged_df, "KNN")

            # Small fix to make sim_options string readable in dataframe
            results_df["metric"] = results_df["param_sim_options"].apply(
                lambda x: x["name"]
            )
            results_df["type"] = results_df["param_sim_options"].apply(
                lambda x: "User-Based" if x["user_based"] else "Item-Based"
            )

            user_results = results_df[results_df["type"] == "User-Based"]
            item_results = results_df[results_df["type"] == "Item-Based"]

            best_user_row = user_results.loc[user_results["mean_test_rmse"].idxmin()]
            best_item_row = item_results.loc[item_results["mean_test_rmse"].idxmin()]

            c1, c2 = st.columns(2)
            with c1:
                st.success(
                    f"**Best User-Based RMSE:** {best_user_row['mean_test_rmse']:.6f}"
                )
            with c2:
                st.info(
                    f"**Best Item-Based RMSE:** {best_item_row['mean_test_rmse']:.6f}"
                )

            st.caption("Best parameters:")
            st.json(best_params)

            def highlight_best_types(row):
                if row.name == best_user_row.name:
                    return ["background-color: #378353;"] * len(row)
                elif row.name == best_item_row.name:
                    return ["background-color: #1f6eb5;"] * len(row)
                else:
                    return [""] * len(row)

            st.dataframe(
                results_df[["param_k", "type", "metric", "mean_test_rmse"]]
                .sort_values("mean_test_rmse")
                .style.apply(highlight_best_types, axis=1),
                width="stretch",
            )

with st.expander("Phase 5: Baseline Models", expanded=False):
    # Global Mean baseline
    st.subheader("Global Mean Baseline")
    mean_rating = train_df["Rating"].mean()
    test_df["MeanPrediction"] = mean_rating
    baseline_rmse = np.sqrt(
        mean_squared_error(test_df["Rating"], test_df["MeanPrediction"])
    )
    st.metric("Training Set Mean Rating", f"{mean_rating:.4f}")

    # Popularity baseline
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

        # Checks intersection between "Top 10 Popular" and "What User Liked" (Rating >= 4.0)
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
    st.dataframe(top_10_popular[["Title", "Genres", "RatingCount"]], width="stretch")


with st.expander("Phase 7: Collaborative Filtering (Item-KNN)", expanded=False):
    st.caption("Evaluation of Item-Based KNN vs Baseline.")

    @st.cache_data
    def train_and_predict_item_knn(train_df, test_df):
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(
            train_df[["UserID", "MovieID", "Rating"]], reader
        ).build_full_trainset()

        sim_options = {"name": "pearson", "user_based": False}
        algo_item_knn = KNNBasic(k=20, sim_options=sim_options, verbose=False)
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
        delta=f"{item_knn_rmse - baseline_rmse:.4f} vs Baseline",
        delta_color="inverse",
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

    example_user = selected_user_id
    st.write(f"**Example Prediction for User {example_user}:**")

    user_test_samples = test_df[test_df["UserID"] == example_user].head(5)
    st.dataframe(user_test_samples[["Title", "Rating", "Item_KNN_Prediction"]])

with st.expander("Phase 6: Collaborative Filtering (User-KNN)", expanded=False):
    st.caption("Evaluation of User-Based KNN vs Baseline")

    @st.cache_data
    def train_and_predict_knn(train_df, test_df):
        # Re-create the Surprise objects inside (so Streamlit hashes the DataFrames correctly)
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(
            train_df[["UserID", "MovieID", "Rating"]], reader
        ).build_full_trainset()

        sim_options = {"name": "pearson", "user_based": True}
        algo_knn = KNNBasic(k=20, sim_options=sim_options, verbose=False)
        algo_knn.fit(train_data)

        test_set = list(zip(test_df["UserID"], test_df["MovieID"], test_df["Rating"]))
        predictions = algo_knn.test(test_set)

        return predictions

    with st.spinner("Training KNN & Generating Predictions..."):
        user_knn_predictions = train_and_predict_knn(train_df, test_df)
        test_df["User_KNN_Prediction"] = [pred.est for pred in user_knn_predictions]

    user_knn_rmse = np.sqrt(
        mean_squared_error(test_df["Rating"], test_df["User_KNN_Prediction"])
    )
    user_knn_prec, user_knn_rec = calculate_ranking_metrics(
        user_knn_predictions, 10, 4.0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "User-KNN RMSE (Lower is better)",
        f"{user_knn_rmse:.4f}",
        delta=f"{user_knn_rmse - baseline_rmse:.4f} vs Baseline",
        delta_color="inverse",
    )

    delta_prec = (
        f"{user_knn_prec - pop_prec:.4f} vs Baseline"
        if "pop_prec" in locals()
        else None
    )
    c2.metric(
        "Precision@10 (Higher is better)", f"{user_knn_prec:.4f}", delta=delta_prec
    )
    delta_rec = (
        f"{user_knn_rec - pop_rec:.4f} vs Baseline" if "pop_rec" in locals() else None
    )
    c3.metric("Recall@10 (Higher is better)", f"{user_knn_rec:.4f}", delta=delta_rec)

    if "item_knn_rmse" in locals():
        st.caption("Comparison vs Item-Based KNN")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="RMSE",
            value=f"{user_knn_rmse - item_knn_rmse:.4f}",
            delta=f"{((user_knn_rmse - item_knn_rmse) / item_knn_rmse) * 100:.2f}%",
            delta_color="inverse",
        )
        col2.metric(
            label="Precision@10",
            value=f"{user_knn_prec - item_knn_prec:.4f}",
            delta=f"{((user_knn_prec - item_knn_prec) / item_knn_prec) * 100:.2f}%",
        )
        col3.metric(
            label="Precision@10",
            value=f"{user_knn_rec - item_knn_rec:.4f}",
            delta=f"{((user_knn_rec - item_knn_rec) / item_knn_rec) * 100:.2f}%",
        )

    example_user = selected_user_id
    st.write(f"**Example Prediction for User {example_user}:**")

    user_test_samples = test_df[test_df["UserID"] == example_user].head(5)
    st.dataframe(
        user_test_samples[
            ["Title", "Rating", "User_KNN_Prediction", "Item_KNN_Prediction"]
        ]
    )

with st.expander("Phase 8: Matrix Factorization (SVD)", expanded=True):
    st.caption("Implementing and evaluating SVD")

    @st.cache_data
    def train_and_predict_svd(train_df, test_df):
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(
            train_df[["UserID", "MovieID", "Rating"]], reader
        ).build_full_trainset()

        algo_svd = SVD(n_factors=20, lr_all=0.01, random_state=42)
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
        delta=f"{svd_rmse - baseline_rmse:.4f} vs Baseline",
        delta_color="inverse",
    )

    delta_prec = (
        f"{svd_prec - pop_prec:.4f} vs Baseline" if "pop_prec" in locals() else None
    )
    c2.metric("Precision@10 (Higher is better)", f"{svd_prec:.4f}", delta=delta_prec)
    delta_rec = (
        f"{svd_rec - pop_rec:.4f} vs Baseline" if "pop_rec" in locals() else None
    )
    c3.metric("Recall@10 (Higher is better)", f"{svd_rec:.4f}", delta=delta_rec)

    col1, col2 = st.columns(2)
    if "item_knn_rmse" in locals():
        st.caption("Comparison vs Item-Based KNN")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="RMSE",
            value=f"{svd_rmse - item_knn_rmse:.4f}",
            delta=f"{((svd_rmse - item_knn_rmse) / item_knn_rmse) * 100:.2f}%",
            delta_color="inverse",
        )
        col2.metric(
            label="Precision@10",
            value=f"{svd_prec - item_knn_prec:.4f}",
            delta=f"{((svd_prec - item_knn_prec) / item_knn_prec) * 100:.2f}%",
        )
        col3.metric(
            label="Precision@10",
            value=f"{svd_rec - item_knn_rec:.4f}",
            delta=f"{((svd_rec - item_knn_rec) / item_knn_rec) * 100:.2f}%",
        )
    if "user_knn_rmse" in locals():
        st.caption("Comparison vs User-Based KNN")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="RMSE",
            value=f"{svd_rmse - user_knn_rmse:.4f}",
            delta=f"{((svd_rmse - user_knn_rmse) / user_knn_rmse) * 100:.2f}%",
            delta_color="inverse",
        )
        col2.metric(
            label="Precision@10",
            value=f"{svd_prec - user_knn_prec:.4f}",
            delta=f"{((svd_prec - user_knn_prec) / user_knn_prec) * 100:.2f}%",
        )
        col3.metric(
            label="Precision@10",
            value=f"{svd_rec - user_knn_rec:.4f}",
            delta=f"{((svd_rec - user_knn_rec) / user_knn_rec) * 100:.2f}%",
        )

    if "item_knn_rmse" in locals() and "user_knn_rmse" in locals():
        if svd_rmse < user_knn_rmse and svd_rmse < item_knn_rmse:
            st.success(
                "SVD outperformed Both Item- and User-KNN for RMSE, validating the choice of Matrix Factorization for sparse data."
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
                "User_KNN_Prediction",
                "Item_KNN_Prediction",
            ]
        ]
    )
with st.expander("Phase 9: Demo - Cold Start (Demographic Clustering)", expanded=False):
    st.caption(
        "Since we don't know new user's taste, we group them with similar users (Age/Gender/Job) and recommend what *that group* likes."
    )
    outlier_ids = merged_df[merged_df["Is_Outlier"] == True]["UserID"].unique()
    clean_users_df = users[~users["UserID"].isin(outlier_ids)]
    st.info(
        f"Training Clustering Model on **{len(clean_users_df)}** users (Removed {len(outlier_ids)} outliers to prevent noise)."
    )

    kmeans_model, scaler, clustered_users = train_user_clustering(clean_users_df)

    with st.spinner("Calculating Cluster Predictions..."):
        cluster_preds = get_clustering_predictions(train_df, test_df, clustered_users)

    y_true = [p[2] for p in cluster_preds]  # index 2 is true_r
    y_pred = [p[3] for p in cluster_preds]  # index 3 is est

    clust_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    clust_prec, clust_rec = calculate_ranking_metrics(
        cluster_preds, k=10, threshold=4.0
    )

    c1, c2, c3 = st.columns(3)

    # RMSE vs Baseline
    c1.metric(
        "Cluster RMSE (Lower is better)",
        f"{clust_rmse:.4f}",
        delta=f"{clust_rmse - baseline_rmse:.4f} vs Baseline",
        delta_color="inverse",
    )

    # Precision vs Baseline
    delta_prec = (
        f"{clust_prec - pop_prec:.4f} vs Baseline" if "pop_prec" in locals() else None
    )
    c2.metric("Precision@10", f"{clust_prec:.4f}", delta=delta_prec)

    # Recall vs Baseline
    delta_rec = (
        f"{clust_rec - pop_rec:.4f} vs Baseline" if "pop_rec" in locals() else None
    )
    c3.metric("Recall@10", f"{clust_rec:.4f}", delta=delta_rec)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        new_gender = st.selectbox("Gender", ["Male", "Female"])
        gender_code = 1 if new_gender == "Male" else 0
    with col2:
        age_map = {
            "Under 18": 1,
            "18-24": 18,
            "25-34": 25,
            "35-44": 35,
            "45-49": 45,
            "50-55": 50,
            "56+": 56,
        }
        new_age_str = st.selectbox("Age Group", list(age_map.keys()), index=2)
        age_code = age_map[new_age_str]
    with col3:
        occ_map = {
            "Student": 4,
            "Academic/Educator": 1,
            "Executive/Manager": 0,
            "Writer/Artist": 20,
            "Programmer/Engineer": 12,
            "Other": 7,
        }
        new_occ_str = st.selectbox("Occupation", list(occ_map.keys()))
        occ_code = occ_map[new_occ_str]

    # Predict cluster on the manual input
    input_df = pd.DataFrame(
        [[gender_code, age_code, occ_code]],
        columns=["Gender_Code", "Age", "Occupation"],
    )
    input_features = scaler.transform(input_df)
    # input_features = scaler.transform([[gender_code, age_code, occ_code]])
    predicted_cluster = kmeans_model.predict(input_features)[0]

    st.success(f"**Result:** This profile falls into **Cluster {predicted_cluster}**.")

    # Show cold-start
    cluster_recs = get_cluster_top_movies(
        predicted_cluster, clustered_users, train_df, movies
    )

    st.dataframe(
        cluster_recs[["Title", "Genres", "MeanRating", "RatingCount"]].style.format(
            {"MeanRating": "{:.2f}"}
        ),
        hide_index=True,
        width="stretch",
    )
