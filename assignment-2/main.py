# Emil Hansson (emhs21@student.bth.se)
# ------------------------------------
# 3 supervised classification algorithms:
#   - LogisticRegression
#   - RandomForestClassifier
#   - SVC

import time
import math
from typing import Any, List, Tuple
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats


# Printable constants (for a more readable terminal)
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

FEATURE_NAMES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]


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


def load_spam_dataset(
    data_path: str = "data/spambase.data",
) -> tuple[NDArray[Any], NDArray[Any], pd.DataFrame]:
    cols = FEATURE_NAMES + ["class"]

    df = pd.read_csv(data_path, header=None, names=cols)

    X = np.array(df[FEATURE_NAMES].values)
    y = np.array(df["class"].values.astype(int))

    print("df shape:", df.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(df.head())

    return X, y, df


def stratified_k_fold(
    X: NDArray[Any],
    y: NDArray[Any],
    algorithms: List[Tuple[str, Any]],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    algo_len = len(algorithms)

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    train_times = np.zeros((n_splits, algo_len))
    accs = np.zeros((n_splits, algo_len))
    f1s = np.zeros((n_splits, algo_len))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=0):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for j, (name, clf) in enumerate(algorithms):
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            t1 = time.perf_counter()
            train_times[fold_idx, j] = t1 - t0
            y_pred = clf.predict(X_test)
            accs[fold_idx, j] = accuracy_score(y_test, y_pred)
            f1s[fold_idx, j] = f1_score(y_test, y_pred, zero_division="warn")

    return train_times, accs, f1s


def summarize_table(
    values: NDArray[Any], metric_name: str, algorithms: List[Tuple[str, Any]]
):
    # print per-fold results like Example 12.4: each row is a fold, last rows avg and stdev
    df_res = pd.DataFrame(values, columns=pd.Index([name for name, _ in algorithms]))
    df_res.index = [f"Fold {i + 1}" for i in range(values.shape[0])]
    avg = df_res.mean()
    std = df_res.std(
        ddof=0
    )  # Example uses population stdev (ddof=0) in the book tables
    df_res.loc["avg"] = avg
    df_res.loc["stdev"] = std
    # Round nicely for display
    return df_res.round(4)


def friedman_test(ranking_matrix, algorithms, metric_name, display_matrix=None):
    """
    ranking_matrix: used to compute ranks (higher -> rank 1). shape (n_folds, k)
    display_matrix: numeric values to print for each fold (same shape). If None, ranking_matrix is used.
    algorithms: list of (name, model) pairs
    """
    if display_matrix is None:
        display_matrix = ranking_matrix

    n, k = display_matrix.shape
    algo_names = [name for name, _ in algorithms]

    # compute ranks from ranking_matrix (best -> rank 1)
    ranks = np.zeros_like(ranking_matrix)
    for i in range(n):
        order = np.argsort(-ranking_matrix[i, :])  # descending: best gets rank 1
        ranks[i, order] = np.arange(1, k + 1)
    avg_ranks = ranks.mean(axis=0)

    print("=" * 80)
    print(f"Table: Friedman Test - Algorithm Rankings for {metric_name}")
    print("=" * 80)
    print()

    # header
    header = "Fold".ljust(13)
    for name in algo_names:
        header += name.ljust(28)
    print(header)
    print("-" * 80)

    # fold rows: show display value and rank in parentheses
    for i in range(n):
        row = f"{i + 1}".ljust(13)
        for j in range(k):
            val = display_matrix[i, j]
            rank = int(ranks[i, j])
            row += f"{val:.4f}({rank})".ljust(28)
        print(row)

    print("-" * 80)
    avg_row = "avg_rank".ljust(13)
    for j in range(k):
        avg_row += f"{avg_ranks[j]:.2f}".ljust(28)
    print(avg_row)
    print()
    print("-" * 80)

    # Friedman chi2 & p-value based on ranking_matrix (ranks already derived)
    # R = (k + 1) / 2
    # chi2 = (12 * n) / (k * (k + 1)) * np.sum((avg_ranks - R) ** 2)
    # p_value = stats.chi2.sf(chi2, df=k - 1)  # = 1 - cdf
    # print("Friedman Test Statistics:")
    # print(f"Chi-square statistic (X^2_F): {chi2:.4f}")
    # print(f"Degrees of freedom: {k - 1}")
    # print(f"P-value: {p_value:.6f}")


def run_nemenyi_test(avg_ranks, n, k, algo_names):
    q_alpha_005 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }

    q_val = q_alpha_005.get(k)
    if not q_val:
        print(f"Warning: Critical value for k={k} not found. Using k=3 value.")
        q_val = 2.343

    # Critical difference
    cd = q_val * np.sqrt((k * (k + 1)) / (6 * n))

    print("\n" + "=" * 80)
    print("Nemenyi (alpha=0.05)")
    print(f"Critical Difference (CD) = {cd:.4f}")
    print("=" * 80)

    significant_found = False
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            is_sig = diff > cd

            sig_str = f"{BOLD}SIGNIFICANT{RESET}" if is_sig else "Not Significant"
            if is_sig:
                significant_found = True

            print(
                f"{algo_names[i]:<20} vs {algo_names[j]:<20} | Diff: {diff:.4f} | {sig_str}"
            )

    if not significant_found:
        print("\nResult: No algorithm pairs are significantly different.")
    else:
        print(
            f"\nResult: Algorithms with rank difference > {cd:.4f} are significantly different."
        )


def main():
    print("assignment-2!")
    X, y, df = load_spam_dataset(data_path="data/spambase.data")

    # Check some statistics:
    n_total = len(y)
    n_spam = np.sum(y == 1)
    percent_spam = (n_spam / n_total) * 100

    print(f"Total samples: {n_total}")
    print(f"Spam samples: {n_spam}")
    print(f"Spam percentage: {percent_spam:.2f}%")
    # -------------------------

    algorithms = [
        (
            "LogisticRegression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000
                ),  # Use scaler so penalty is applied evenly
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=200, n_jobs=-1),
        ),
        (
            "SVM",
            make_pipeline(
                StandardScaler(),
                SVC(
                    probability=True, gamma="scale"
                ),  # Using scaler because its distance based
            ),
        ),
    ]
    # Procedure:
    # 1. Run stratified ten-fold cross-validation tests:
    train_times, accs, f1s = stratified_k_fold(X=X, y=y, algorithms=algorithms)

    for metric_name, matrix, larger_is_better in [
        ("train_time", train_times, False),
        ("accuracy", accs, True),
        ("f1", f1s, True),
    ]:
        print(
            f"\n=== {metric_name} per fold {'(seconds)' if metric_name == 'train_time' else ''} ==="
        )
        print(summarize_table(matrix, metric_name, algorithms).to_string())

        if larger_is_better:
            ranking_matrix = matrix.copy()
        else:
            ranking_matrix = -matrix.copy()

        print(f"\n=== Friedman test for {metric_name} ===")
        # print table using original display values but ranks computed from ranking_matrix
        friedman_test(
            ranking_matrix, algorithms, metric_name=metric_name, display_matrix=matrix
        )

        # Calculate chi2 and p-value to pass to Nemenyi logic
        n, k = ranking_matrix.shape
        ranks = np.zeros_like(ranking_matrix)
        for i in range(n):
            order = np.argsort(-ranking_matrix[i, :])
            ranks[i, order] = np.arange(1, k + 1)

        avg_ranks = ranks.mean(axis=0)
        R = (k + 1) / 2
        chi2 = (12 * n) / (k * (k + 1)) * np.sum((avg_ranks - R) ** 2)
        p_value = stats.chi2.sf(chi2, df=k - 1)
        print("Friedman Test Statistics:")
        print(f"Chi-square statistic (X^2_F): {chi2:.4f}")
        print(f"Degrees of freedom: {k - 1}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            print(
                f"\n{GREEN}Friedman test significant (p < 0.05). Proceeding to Nemenyi test.{RESET}"
            )
            algo_names = [name for name, _ in algorithms]
            run_nemenyi_test(avg_ranks, n, k, algo_names)
        else:
            print(
                f"\n{YELLOW}Friedman test NOT significant (p >= 0.05). Skipping Nemenyi.{RESET}"
            )


if __name__ == "__main__":
    main()
