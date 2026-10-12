import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("sklearn-classifier")

# hyperparameters (inhe badal kar nayi runs banao)
params = {
    "n_estimators": 50,
    "max_depth": 3,
    "min_samples_split": 2,
    "random_state": 42
}


with mlflow.start_run():
    # --- data ---
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    mlflow.log_param("dataset", "breast_cancer")
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))

    # --- hyperparameters log karo ---
    mlflow.log_params(params)

    # --- train ---
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # --- evaluate ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    mlflow.log_metrics(metrics)

    # --- feature importance plot ---
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([data.feature_names[i] for i in top_idx], importances[top_idx])
    ax.set_title("Top 10 feature importances")
    fig.tight_layout()
    fig.savefig("feature_importance.png", dpi=100)
    mlflow.log_artifact("feature_importance.png", artifact_path="plots")
    plt.close()

    # --- model log karo ---
    mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})