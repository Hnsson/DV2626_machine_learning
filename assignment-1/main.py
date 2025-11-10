import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# import dataset
df = pd.read_csv("data/winequality-white.csv", sep=";")

# Separate features (X) and target (y)
X = df.drop(columns=["quality"])  # all columns except target variable
y = df["quality"]

# ------------- Step 1: Inspect dataset -------------
print(df.info())
print(df.describe())

counts = df["quality"].value_counts().sort_index()
ratios = counts / sum(counts)
print(ratios)


# ------------- Step 2: Divide the data into train and test sets. -------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,  # wanted to keep reproducability for assignemnt
)

# ------------- Step 3: Perform scaling on the data. -------------
#         I'm going to perform standardization and transform all
#         numeric features so that they're on a similar numerical range.
#         Usually wants a mean zero and a standard deviation 1
#         (there also exists min-max scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Mean after scaling:", np.mean(X_train_scaled, axis=0))  # mean is near zero
print("Std after scaling:", np.std(X_train_scaled, axis=0))  # standard deviation is 1

# ------------- Step 4: Repeated Stratified k-Fold CV on the TRAIN set -------------
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
}

chosen_scoring = "f1_weighted"  # Also saw f1_macro?

cv_results = {}
for name, model in models.items():
    pipeline = Pipeline(
        [("smote", SMOTE(random_state=42, k_neighbors=1)), ("model", model)]
    )
    scores = cross_val_score(
        pipeline, X_train_scaled, y_train, cv=cv, scoring=chosen_scoring, n_jobs=-1
    )
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name} {chosen_scoring}: {scores.mean():.4f} +- {scores.std():.4f}")

best_name = max(cv_results, key=lambda k: cv_results[k][0])
print(
    f"\nBest by {chosen_scoring}: {best_name} → {cv_results[best_name][0]:.4f} +- {cv_results[best_name][1]:.4f}"
)  # RandomForest was best


# ------------- Step 5: Fit the best model on the WHOLE training set & evaluate on TEST -------------
best_model = models[best_name]
best_model.fit(X_train_scaled, y_train)


# ------------- Step 6: Model's performance on the test set. -------------
y_pred = best_model.predict(X_test_scaled)

print("\nTest set performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification report:\n", classification_report(y_test, y_pred))


# ------------- Step 7: Balance the SCALED TRAIN set with SMOTE -------------
print("Original training class distribution:")
print(Counter(y_train))

smote = SMOTE(random_state=42, k_neighbors=2)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("\nBalanced training class distribution (after SMOTE):")
print(Counter(y_train_bal))

# ------------- Step 8: Repeat CV on the BALANCED training set -------------
cv_results_bal = {}
for name, model in models.items():
    pipeline = Pipeline(
        [("smote", SMOTE(random_state=42, k_neighbors=1)), ("model", model)]
    )
    scores = cross_val_score(
        pipeline,
        X_train_bal,
        y_train_bal,
        cv=cv,
        scoring=chosen_scoring,
        n_jobs=-1,
    )
    cv_results_bal[name] = (scores.mean(), scores.std())
    print(
        f"{name} {chosen_scoring} (balanced): {scores.mean():.4f} +- {scores.std():.4f}"
    )

best_name_bal = max(cv_results_bal, key=lambda k: cv_results_bal[k][0])
print(
    f"\nBest by {chosen_scoring} (balanced): {best_name_bal} → "
    f"{cv_results_bal[best_name_bal][0]:.4f} +- {cv_results_bal[best_name_bal][1]:.4f}"
)

# ------------- Step 9: Train the best model on the BALANCED train set & evaluate on the SAME test set -------------
best_model_bal = models[best_name_bal]
best_model_bal.fit(X_train_bal, y_train_bal)

y_pred_bal = best_model_bal.predict(X_test_scaled)

print("\n[Step 9] Test set performance (after training on balanced data):")
print("Accuracy (balanced):", accuracy_score(y_test, y_pred_bal))
print("Macro-F1 (balanced):", f1_score(y_test, y_pred_bal, average="macro"))
print(
    "\nClassification report (balanced):\n",
    classification_report(y_test, y_pred_bal, zero_division=0),
)


# ------------- Step 10: Discuss your findings -------------
print("VAAAAAS?")
