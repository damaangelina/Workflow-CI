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

# ======================================================
# MLFLOW SETUP 
# ======================================================
if os.getenv("GITHUB_ACTIONS") != "true":
    try:
        import dagshub
        dagshub.init(
            repo_owner="damaangelina",
            repo_name="Eksperimen_SML_NiKomangDamaAngelina",
            mlflow=True
        )
    except Exception:
        pass 
else:
    mlflow.set_tracking_uri("file:///tmp/mlruns")

mlflow.set_experiment("Model_Tuning_Experiment")

# ======================================================
# LOAD DATA
# ======================================================
data_path = "cancer_preprocessing/breast_cancer_preprocessing.csv"
df = pd.read_csv(data_path)

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
    # Metrics
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
    # Save Model
    # =====================
    mlflow.sklearn.log_model(best_model, "model")

    # ==================================================
    # ARTEFAK
    # ==================================================
    os.makedirs("artifacts", exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("artifacts/training_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("artifacts/training_confusion_matrix.png")

    # Metric Info
    metric_info = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    with open("artifacts/metric_info.json", "w") as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact("artifacts/metric_info.json")

    # Estimator
    with open("artifacts/estimator.html", "w") as f:
        f.write(f"<pre>{best_model}</pre>")
    mlflow.log_artifact("artifacts/estimator.html")

    # Classification Report
    with open("artifacts/classification_report.json", "w") as f:
        json.dump(classification_report(y_test, y_pred, output_dict=True), f, indent=4)
    mlflow.log_artifact("artifacts/classification_report.json")

    # Train-Test Info
    train_test_info = {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    with open("artifacts/train_test_info.json", "w") as f:
        json.dump(train_test_info, f, indent=4)
    mlflow.log_artifact("artifacts/train_test_info.json")

    print("Training selesai.")
    print("Best parameters:", grid_search.best_params_)
    print(f"Accuracy: {acc:.4f}")
