import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_all_data

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
st.title("Exploratory Data Analysis (EDA)")

ratings, users, movies = load_all_data()

tab1, tab2, tab3 = st.tabs(["Ratings Analysis", "Demographics", "Correlations"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Distribution of Ratings")

        rating_counts = ratings["Rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]

        fig_ratings = px.bar(
            rating_counts,
            x="Rating",
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig_ratings.update_layout(
            xaxis_title="Star Rating", yaxis_title="Number of Votes"
        )
        st.plotly_chart(fig_ratings, use_container_width=True)
    with col2:
        st.subheader("Sparsity Metric")
        n_users = int(users["UserID"].nunique())
        n_movies = int(movies["MovieID"].nunique())

        total_actual = len(ratings)
        total_possible = n_users * n_movies
        sparsity = 1 - (total_actual / total_possible)

        st.metric("Total Users", f"{n_users:,}")
        st.metric("Total Movies", f"{n_movies:,}")
        st.metric("Sparsity", f"{sparsity:.2%}")
        st.info("High sparsity justifies the use of Matrix Factorization (SVD).")
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        age_counts = users["AgeDesc"].value_counts().reset_index()
        age_counts.columns = ["Age Group", "Count"]

        age_order = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]

        fig_age = px.bar(
            age_counts,
            x="Age Group",
            y="Count",
            text="Count",
            category_orders={"Age Group": age_order},
            color="Count",
            color_continuous_scale="Magma",
        )
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        st.subheader("Occupation Distribution")
        occ_counts = users["OccupationDesc"].value_counts().reset_index()
        occ_counts.columns = ["Occupation", "Count"]

        fig_occ = px.bar(
            occ_counts,
            y="Occupation",
            x="Count",
            orientation="h",
            title="Users by Occupation",
            color="Count",
            height=600,
        )
        fig_occ.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_occ, use_container_width=True)

    st.divider()
    st.subheader("Edge Group Analysis (Under 18 vs 56+)")

    user_activity = (
        ratings.groupby("UserID")
        .agg(TotalRatings=("Rating", "count"), AvgRating=("Rating", "mean"))
        .reset_index()
    )
    active_users = pd.merge(users, user_activity, on="UserID")
    edge_groups = active_users[active_users["AgeDesc"].isin(["Under 18", "56+"])]

    outlier_col1, outlier_col2 = st.columns([1, 2])

    with outlier_col1:
        st.markdown("**High Volume Users (>500 ratings)**")
        outliers = edge_groups[edge_groups["TotalRatings"] > 500].sort_values(
            "TotalRatings", ascending=False
        )
        st.dataframe(
            outliers[["UserID", "AgeDesc", "TotalRatings", "AvgRating"]],
            use_container_width=True,
            hide_index=True,
            height=300,
        )

    with outlier_col2:
        st.markdown("**Rating Distribution Spread**")
        fig_box = px.box(
            edge_groups,
            x="AgeDesc",
            y="AvgRating",
            color="AgeDesc",
            points="all",
            category_orders={"AgeDesc": ["Under 18", "56+"]},
        )
        fig_box.update_layout(
            margin=dict(t=10, b=0, l=0, r=0), height=300, showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
with tab3:
    merged_df = pd.merge(ratings, users, on="UserID")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Rating by Gender")
        avg_gender = merged_df.groupby("Gender")["Rating"].mean().reset_index()

        fig_gen = px.bar(
            avg_gender,
            x="Gender",
            y="Rating",
            color="Gender",
            text_auto=".2f",
            title="Do men or women rate higher?",
        )
        st.plotly_chart(fig_gen, use_container_width=True)
    with col2:
        st.subheader("Average Rating by Age Group")
        avg_age = merged_df.groupby("AgeDesc")["Rating"].mean().reset_index()

        age_order = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]

        avg_age["AgeDesc"] = pd.Categorical(
            avg_age["AgeDesc"], categories=age_order, ordered=True
        )
        avg_age = avg_age.sort_values("AgeDesc")

        fig_age_trend = px.line(
            avg_age,
            x="AgeDesc",
            y="Rating",
            markers=True,
            category_orders={"Age Group": age_order},
            title="Rating Tendency by Age",
        )
        st.plotly_chart(fig_age_trend, use_container_width=True)
