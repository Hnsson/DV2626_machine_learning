import streamlit as st
import pandas as pd

st.set_page_config(page_title="Movie Recommender Group 31", layout="wide")

st.title("Movie Recommender System")
st.caption("Group 31 - Emil Hansson")


@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv(
            "data/ratings.dat",
            sep="::",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            engine="python",
        )
        return ratings
    except FileNotFoundError:
        return None


st.sidebar.header("Settings")
show_raw = st.sidebar.checkbox("Show Raw Data", True)

data = load_data()

if data is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", int(data["UserID"].nunique()))
    col2.metric("Total Movies", int(data["MovieID"].nunique()))
    col3.metric("Total Ratings", data.shape[0])

    if show_raw:
        st.subheader("Raw Data Sample")
        st.caption("First 10 samples")
        st.dataframe(data.head(10))

else:
    st.error(
        "Could not find 'data/ratings.dat'. Please create a 'data' folder and put the MovieLens files inside."
    )

# 5. Placeholder for your Models (Future)
st.divider()
# st.subheader("Recommendation Engine")
# st.info("Run the algorithms from the sidebar (Coming Soon)")
