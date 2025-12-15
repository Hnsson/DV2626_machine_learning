# Project Plan: Hybrid Movie Recommendation
### Phase 1: Project Structure & Frontend
- [ ] Create and setup the appropriate python scripts
- [ ] Include and setup `streamlit` to be used to visualize the ML project easier and display the movie recommendation results easier.

### Phase 2: Data Loading
- [ ] Download MovieLens 1M.
- [ ] Write script to load `ratings.dat`, `users.dat`, and `movies.dat` using Pandas (delimiter is `::`)
- [ ] Merge datasets into a single master DataFrame for inital inspection
- [ ] Check for null values, duplicates, or corrupted lines

### Phase 3: Exploratory Data Analysis (EDA)
*Goal: Understand the data and generate plots.*

**Rating analysis**
- [ ] Plot the distribution of ratings (Are they mostly 4s and 5s?).
- [ ] Calculate "Sparsity" (what % of the user-item matrix is empty?)
- [ ] Plot the "Long tail" (movie popularity distribution)

**User Demographic Analysis**
- [ ] Plot Age distribution
- [ ] Plot Occupation distribution

**Correlation Checks**
- [ ] Check if certain demographics tend to rate higher/lower on average

### Phase 4: Pre-processing & Splitting
- [ ] Perform a split (80% Train, 20% Test)
- [ ] Ensure the split is stratified by user (so every user appears in both sets)
- [ ] Create the **Utility Matrix** (users as rows, movies as cols, ratings as values). Fill `NaN` with 0 for sparse matrix operations (DO NOT train on zeros for SVD)

### Phase 5: Baseline Models
**Global Mean Baseline**
- [ ] Calculate the average of all ratings in Train set
- [ ] Predict this number for every entry in Test set
- [ ] Calculate RMSE

**Popularity Baseline (Non-personalized)**
- [ ] Rank movies by number of ratings (or damped mean)
- [ ] Generate the "Top 10 Popular" list
- [ ] Calculate Precision@10 (how many of these did the user actually watch/like in the test set?)

### Phase 6: Collaborative Filtering
*Goal: Implement the "Memory-Based" and "Model-Based" algorithms.*

**User-KNN (Memory-Based)**
- [ ] Implement/Call User-Based KNN (using cosine or pearson similarity)
- [ ] Tune `k` (k=20, 40, 60, ...)
- [ ] Generate Predictions and measure RMSE.

**Item-KNN (Memory-Based)**
- [ ] Implement/Call Item-Based KNN
- [ ] Compare training speed vs. User-KNN

**Matrix Factorization SVD (Model-Based)**
- [ ] Implement SVD (using `Suprise` library is standard)
- [ ] Tune hyperparamters: `n_factors` (latent factors), `learning_rate`, `regularization`
- [ ] Train on the full training set
- [ ] Extract the "Item Matrices" and see if you can find patterns (do specific latent factors correspond to "Horror" movies?)

### Phase 7: Demographic Clustering (The Hybrid/Cold-Start)
*Goal: Solve the problem where CF fails.*

**Feature Engineering**
- [ ] One-Hot Encode categorical user features (Gender, Occupation)
- [ ] Normalize/Scale Age

**K-Means Clustering**
- [ ] use the "Elbow Method" to find the optimal number of clusters (k)
- [ ] Run K-Means on the user profile data
- [ ] Assign every user to a cluster id

**Cluster-Based Recommendation**
- [ ] For each cluster id, calculate the Top-N highest-rated movies by users in that cluster
- [ ] Simluate cold start: Pick a user from the test set, pretend you have 0 ratings for them, and user their cluster recommendation. Measure if it is better than the "Popularity Baseline".

### Phase 8: Evaluation & Comparison

**Unified Evaluation Script**
- [ ] Write a script that runs all models (Baseline, KNN, SVD, Clustering) on the Test set
- [ ] Calculate Metrics: RMSE / MAE (for rating accuracy), Precision@k / Recall@k (for ranking relevance), NDCG (for ranking order)

**Result Compilation**
- [ ] Create a comparison table: Model Name vs. Metric Score (like assignment 2)
- [ ] Generate a Bar Chart comparing the performance
