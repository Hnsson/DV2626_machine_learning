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

# Printable constants (for a more readable terminal)
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def divider(title: str = "", width: int = 70, color: str = CYAN):
    """
    Prints a fancy colored divider with centered text.
    Example:
    divider("Step 4: Cross Validation")
    """
    RESET = "\033[0m"
    if title:
        title = f" {title} "
    line = title.center(width, "=")
    print(f"\n{color}{line}{RESET}")


# import dataset
df = pd.read_csv("data/winequality-white.csv", sep=";")

# Separate features (X) and target (y)
X = df.drop(columns=["quality"])  # all columns except target variable
y = df["quality"]

# ------------- Step 1: Inspect dataset -------------
divider("Inspect dataset")
counts = y.value_counts().sort_index()
ratios = counts / counts.sum()
print(f"{YELLOW}{BOLD}Class counts:{RESET}\n", counts)
print(f"\n{YELLOW}{BOLD}Class ratios:{RESET}\n", ratios.round(4))

# ------------- Step 2: Divide the data into train and test sets. -------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
)

# ------------- Step 3,4,5: Repeated Stratified k-Fold CV on the TRAIN set -------------
divider("Repeated Stratified k-Fold DV")
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300),
}

scoring = "f1_weighted"  # Also saw f1_macro?

cv_results_unbal = {}
for name, model in models.items():
    pipeline_unbal = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(
        pipeline_unbal, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )
    cv_results_unbal[name] = (scores.mean(), scores.std())
    print(f"{YELLOW}{name} {scoring}: {RESET}{scores.mean():.4f} +- {scores.std():.4f}")

best_unbal = max(cv_results_unbal, key=lambda k: cv_results_unbal[k][0])
print(
    f"{GREEN}{BOLD}Best (unbalanced): {best_unbal}: {RESET}{cv_results_unbal[best_unbal][0]:.4f} +- {cv_results_unbal[best_unbal][1]:.4f}"
)

final_unbal = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", models[best_unbal]),
    ]
).fit(X_train, y_train)

# ------------- Step 6: Model's performance on the test set. -------------
divider("Model's performance on the test set")
y_pred = final_unbal.predict(X_test)

print("Unbalanced training:")
print(f"{YELLOW}{BOLD}Accuracy:{RESET}", accuracy_score(y_test, y_pred))
print(f"{YELLOW}{BOLD}Macro-F1:{RESET}", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred, digits=4, zero_division=0))


# ------------- Step 7: Balance the SCALED TRAIN set with SMOTE -------------
divider("Balance the SCALED TRAIN set with SMOTE")
print(f"{YELLOW}Original training class distribution:{RESET}")
print(Counter(y_train), "\n")

cv_results_bal = {}
for name, model in models.items():
    pipeline_bal = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(k_neighbors=1)),
            ("model", model),
        ]
    )
    scores = cross_val_score(
        pipeline_bal, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )
    cv_results_bal[name] = (scores.mean(), scores.std())
    print(
        f"{YELLOW}[Balanced] {name} {scoring}: {RESET}{scores.mean():.4f} +- {scores.std():.4f}"
    )

best_bal = max(cv_results_bal, key=lambda k: cv_results_bal[k][0])
print(
    f"{GREEN}{BOLD}Best (balanced): {best_bal}: {RESET}{cv_results_bal[best_bal][0]:.4f} +- {cv_results_bal[best_bal][1]:.4f}"
)


# ------------- Step 9: Train the best model on the BALANCED train set & evaluate on the SAME test set -------------
divider("Train best model on BALANCED & Evaluate")
final_bal = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("smote", SMOTE(k_neighbors=1)),
        ("model", models[best_bal]),
    ]
).fit(X_train, y_train)

y_pred_bal = final_bal.predict(X_test)
print("Balanced training (SMOTE in pipeline):")
print(f"{YELLOW}{BOLD}Accuracy:{RESET}", accuracy_score(y_test, y_pred_bal))
print(f"{YELLOW}{BOLD}Macro-F1:{RESET}", f1_score(y_test, y_pred_bal, average="macro"))
print(classification_report(y_test, y_pred_bal, digits=4, zero_division=0))
