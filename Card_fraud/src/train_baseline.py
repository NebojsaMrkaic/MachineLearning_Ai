import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
from preprocess import load_data, preprocess

def main():
    # 1. Load dataset
    df = load_data()

    # ✅ FULL DATASET: no sampling here
    # The dataset has ~284,807 rows. This will take longer but gives realistic results.
    # (Remove the sample line used earlier.)

    # 2. Preprocess (split + scale)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = preprocess(df)

    # 3. Define models
    models = [
        (LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1), "Logistic Regression"),
        # ⚠️ KNN is very slow on full dataset. Keep only if you want to compare.
        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "KNN"),
        (DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42), "Decision Tree"),
        # ✅ Random Forest with more trees for full dataset
        (RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42), "Random Forest"),
        # ✅ Tuned Gradient Boosting for full dataset
        (GradientBoostingClassifier(
            n_estimators=300,      # more trees for stronger learning
            learning_rate=0.05,    # smaller steps for stability
            max_depth=3,           # shallow trees to prevent overfitting
            random_state=42
        ), "Gradient Boosting (Tuned)")
    ]

    results = []
    pr_curves = []
    roc_curves = []

    # 4. Train + evaluate each model
    for model, name in models:
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)

        # Predict probabilities
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]

        # Metrics
        val_auprc = average_precision_score(y_val, val_probs)
        val_roc = roc_auc_score(y_val, val_probs)
        test_auprc = average_precision_score(y_test, test_probs)
        test_roc = roc_auc_score(y_test, test_probs)

        results.append({
            "Model": name,
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

    # 5. Print comparison table
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison (full dataset) ===")
    print(results_df.to_string(index=False))

    # 6. Summarized PR curve plot
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

    # 7. Summarized ROC curve plot
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
