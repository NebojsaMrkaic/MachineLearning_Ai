import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
from xgboost import XGBClassifier
import sys, os

# ✅ Force Python to use your local src/ folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")   # adjust if preprocess.py is elsewhere
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from preprocess import load_data, preprocess   # now imports your local file, not site-packages


def main():
    # 1. Load dataset
    df = load_data()

    # ✅ FULL DATASET
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = preprocess(df)

    # 2. Define models
    models = [
        (LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1),
         "Logistic Regression"),
        (XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"   # ✅ no warning, no label encoder
        ), "XGBoost")
    ]

    results = []
    pr_curves = []
    roc_curves = []

    # 3. Train + evaluate each model
    for model, name in models:
        print(f"[INFO] Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict probabilities
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]

        # Measure inference time per transaction
        start_pred = time.time()
        _ = model.predict_proba(X_test[:1])[:, 1]  # single transaction
        single_pred_time = time.time() - start_pred

        # Metrics
        val_auprc = average_precision_score(y_val, val_probs)
        val_roc = roc_auc_score(y_val, val_probs)
        test_auprc = average_precision_score(y_test, test_probs)
        test_roc = roc_auc_score(y_test, test_probs)

        results.append({
            "Model": name,
            "Train Time (s)": train_time,
            "Pred Time (s/transaction)": single_pred_time,
            "Val AUPRC": val_auprc,
            "Val ROC_AUC": val_roc,
            "Test AUPRC": test_auprc,
            "Test ROC_AUC": test_roc
        })

        # Store curves for plotting later
        precisions, recalls, _ = precision_recall_curve(y_test, test_probs)
        pr_curves.append((recalls, precisions, name, test_auprc))

        fpr, tpr, _ = roc_curve(y_test, test_probs)
        roc_curves.append((fpr, tpr, name, test_roc))

    # 4. Print comparison table
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison (full dataset) ===")
    print(results_df.to_string(index=False))

    # 5. Summarized PR curve plot
    plt.figure(figsize=(7, 5))
    for recalls, precisions, name, auprc in pr_curves:
        plt.plot(recalls, precisions, label=f"{name} (AUPRC={auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (Test Set, Full Data)")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # legend outside
    plt.tight_layout()
    plt.show()

    # 6. Summarized ROC curve plot
    plt.figure(figsize=(7, 5))
    for fpr, tpr, name, roc in roc_curves:
        plt.plot(fpr, tpr, label=f"{name} (ROC_AUC={roc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Test Set, Full Data)")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # legend outside
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
