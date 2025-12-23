import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path

# ======================================================
# MLFLOW SETUP
# ======================================================
mlflow.set_experiment("Model_Tuning_Experiment")

# ======================================================
# LOAD DATA 
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cancer_processing" / "processed_cancer.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan di path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================================
# TRAINING + TUNING
# ======================================================
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run(run_name="model_tuning_run"):
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # =====================
    # METRICS
    # =====================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # =====================
    # SAVE MODEL
    # =====================
    mlflow.sklearn.log_model(best_model, "model")

    # ==================================================
    # ARTEFAK
    # ==================================================
    ARTIFACT_DIR = BASE_DIR / "artifacts"
    ARTIFACT_DIR.mkdir(exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = ARTIFACT_DIR / "training_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(str(cm_path))

    # 2. Metric Info
    metric_info = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    metric_path = ARTIFACT_DIR / "metric_info.json"
    with open(metric_path, "w") as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact(str(metric_path))

    # 3. Estimator
    estimator_path = ARTIFACT_DIR / "estimator.html"
    with open(estimator_path, "w") as f:
        f.write(f"<pre>{best_model}</pre>")
    mlflow.log_artifact(str(estimator_path))

    # 4. Classification Report
    report_path = ARTIFACT_DIR / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(
            classification_report(y_test, y_pred, output_dict=True),
            f,
            indent=4
        )
    mlflow.log_artifact(str(report_path))

    # 5. Train-Test Info 
    train_test_info = {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X.shape[1]
    }
    train_test_path = ARTIFACT_DIR / "train_test_info.json"
    with open(train_test_path, "w") as f:
        json.dump(train_test_info, f, indent=4)
    mlflow.log_artifact(str(train_test_path))

    # 6. Model Parameters
    model_params_path = ARTIFACT_DIR / "model_params.json"
    with open(model_params_path, "w") as f:
        json.dump(best_model.get_params(), f, indent=4)
    mlflow.log_artifact(str(model_params_path))

    print("Training selesai.")
    print("Best parameters:", grid_search.best_params_)
    print(f"Accuracy: {acc:.4f}")
