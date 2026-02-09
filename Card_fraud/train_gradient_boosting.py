import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
import sys, os

# ✅ Force Python to look in your project src/ folder first
# Adjust the path if your preprocess.py is not inside "src"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from preprocess import load_data, preprocess   # now imports your local file, not site-packages


def main():
    # 1. Load dataset
    df = load_data()

    # ✅ FULL DATASET: no sampling here
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = preprocess(df)

    # 2. Define upgraded Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=500,       # more trees for stronger learning
        learning_rate=0.01,     # smaller steps for stability
        max_depth=4,            # slightly deeper trees
        subsample=0.8,          # stochastic boosting to reduce overfitting
        min_samples_leaf=20,    # avoid tiny leaves
        random_state=42
    )

    print("[INFO] Training Gradient Boosting...")
    gb.fit(X_train, y_train)

    # 3. Predict probabilities
    val_probs = gb.predict_proba(X_val)[:, 1]
    test_probs = gb.predict_proba(X_test)[:, 1]

    # 4. Metrics
    val_auprc = average_precision_score(y_val, val_probs)
    val_roc = roc_auc_score(y_val, val_probs)
    test_auprc = average_precision_score(y_test, test_probs)
    test_roc = roc_auc_score(y_test, test_probs)

    print("\n=== Gradient Boosting Results (full dataset) ===")
    print(f"Val AUPRC: {val_auprc:.4f}, Val ROC_AUC: {val_roc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}, Test ROC_AUC: {test_roc:.4f}")

    # 5. Precision–Recall curve
    precisions, recalls, _ = precision_recall_curve(y_test, test_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label=f"GB (AUPRC={test_auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Gradient Boosting, Test Set)")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # legend outside
    plt.tight_layout()
    plt.show()

    # 6. ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"GB (ROC_AUC={test_roc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Gradient Boosting, Test Set)")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # legend outside
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
