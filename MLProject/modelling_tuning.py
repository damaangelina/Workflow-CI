import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
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
# 1. MLFLOW DAGSHUB SETUP 
# ======================================================
dagshub.init(
    repo_owner="damaangelina",
    repo_name="Eksperimen_SML_NiKomangDamaAngelina",
    mlflow=True
)

mlflow.set_experiment("Model_Tuning_Experiment")

# ======================================================
# 2. LOAD DATASET 
# ======================================================
data_path = "cancer_preprocessing/breast_cancer_preprocessing.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError("Dataset preprocessing tidak ditemukan!")

df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# 3. HYPERPARAMETER TUNING
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

# ======================================================
# 4. TRAINING + LOGGING (LOCAL)
# ======================================================
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
    # 5. ARTEFAK 
    # ==================================================
    os.makedirs("artifacts", exist_ok=True)

    # training_confusion_matrix.png
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "artifacts/training_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # metric_info.json
    metric_info = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    metric_path = "artifacts/metric_info.json"
    with open(metric_path, "w") as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact(metric_path)

    # estimator.html
    estimator_path = "artifacts/estimator.html"
    with open(estimator_path, "w") as f:
        f.write(
            f"""
            <html>
            <body>
                <h2>RandomForestClassifier</h2>
                <pre>{best_model}</pre>
            </body>
            </html>
            """
        )
    mlflow.log_artifact(estimator_path)

    print("Training selesai.")
    print("Best parameters :", grid_search.best_params_)
    print(f"Accuracy        : {acc:.4f}")

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

    # training_confusion_matrix.png 
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "artifacts/training_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # metric_info.json 
    metric_info = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    metric_path = "artifacts/metric_info.json"
    with open(metric_path, "w") as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact(metric_path)

    # estimator.html 
    estimator_path = "artifacts/estimator.html"
    with open(estimator_path, "w") as f:
        f.write(
            f"""
            <html>
            <body>
                <h2>RandomForestClassifier</h2>
                <pre>{best_model}</pre>
            </body>
            </html>
            """
        )
    mlflow.log_artifact(estimator_path)

    # ==================================================
    # TAMBAHAN 2 ARTEFAK 
    # ==================================================

    # classification_report.json
    class_report = classification_report(
        y_test, y_pred, output_dict=True
    )
    class_report_path = "artifacts/classification_report.json"
    with open(class_report_path, "w") as f:
        json.dump(class_report, f, indent=4)
    mlflow.log_artifact(class_report_path)

    # train_test_info.json
    train_test_info = {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": 0.2,
        "random_state": 42
    }
    tti_path = "artifacts/train_test_info.json"
    with open(tti_path, "w") as f:
        json.dump(train_test_info, f, indent=4)
    mlflow.log_artifact(tti_path)

    print("Training selesai.")
    print("Best parameters :", grid_search.best_params_)
    print(f"Accuracy        : {acc:.4f}")
