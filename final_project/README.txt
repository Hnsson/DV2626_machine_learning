# DV2626 Final Project: Hybrid Movie Recommendation System
Note: Render the README.md markdown file instead for better visuals.

This project implements a **Hybrid Recommender System** using the MovieLens 1M dataset. It benchmarks Memory-Based Collaborative Filtering (User-KNN, Item-KNN) against Model-Based approaches (Matrix Factorization via SVD) and implements a Demographic Clustering solution for the "Cold-Start" problem.

The application is built with **Streamlit** to provide an interactive frontend for data exploration and model testing.

## Installation
You can set up the project using **uv** (recommended/fastest) or standard **pip**.

**Option 1: Use `uv` (recommended)**
If you have `uv` installed, you don't need to manually create a venv.
1. Sync dependencies:
```bash
uv sync
```

**Option 2: Standard python**
If you prefer standard python.
1. Create a venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

##  Running it
To start the streamlit server and view dashboard:
**Using `uv`**
```bash
uv run streamlit run Home.py
```

**Using Standard python**
```bash
streamlit run Home.py
```

The app should automatically open in your browser at `http://localhost:8501`

## Screenshots
### Home page 
Here the home page is just a starting ground for the other pages (EDA and Movie Recommendation)

### Exploratory Data Analysis (EDA)
Here is a page about data analysis where in the top is the sizes of the datasets and then 3 tabs with different data (ratings, demographics, correlations)

### Movie recommendation
Here is the main experiment with all the steps in the project in different expanders starting with pre-processing, then model configuration and then baseline evaluation. Then the CF models Item-KNN, User-KNN and SVD being evaluated and compared with baseline as well as each other. Then lastly is a expander with the demographic clustering.
