# Emil Hansson (emhs21@student.bth.se)
# ------------------------------------
# 3 supervised classification algorithms:
#   - LogisticRegression
#   - RandomForestClassifier
#   - SVC

import numpy as np
import pandas as pd

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


def load_spam_dataset(data_path="data/spambase.data"):
    feature_cols = [f"f{i}" for i in range(57)]
    cols = feature_cols + ["class"]

    df = pd.read_csv(data_path, header=None, names=cols)
    X = df[feature_cols].values
    y = df["class"].values.astype(int)
    print("Loaded dataset with shape:", X.shape)


def main():
    print("assignment-2!")
    load_spam_dataset()


if __name__ == "__main__":
    main()
