import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

EXPERIMENT = "mlops-pipeline"
MODEL_NAME = "ProductionClassifier"
ACCURACY_GATE = 0.93  # promotion ke liye minimum accuracy

TRUSTED = [
    "sklearn.tree._tree.Tree",
]

mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

candidates = [
    ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), False),
    ("GradBoost", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), False),
    ("LogisticReg", LogisticRegression(C=1.0, max_iter=1000, random_state=42), True),
]

run_results = []

for name, model, scaled in candidates:
    Xtr_use = X_tr_s if scaled else X_tr
    Xte_use = X_te_s if scaled else X_te

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("scaled", scaled)
        model.fit(Xtr_use, y_tr)
        acc = accuracy_score(y_te, model.predict(Xte_use))
        auc = roc_auc_score(y_te, model.predict_proba(Xte_use)[:, 1])
        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
        try:
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED)
        except Exception as e:
            print(f"  (model save warning for {name}: {str(e)[:80]})")
        run_results.append({"name": name, "run_id": run.info.run_id, "accuracy": acc, "roc_auc": auc})
        print(f"  {name:15s}  acc={acc:.4f}  auc={auc:.4f}")

# --- automatic promotion gate ---
best = max(run_results, key=lambda r: r["roc_auc"])
print(f"\nBest model: {best['name']} (roc_auc={best['roc_auc']:.4f})")

if best["accuracy"] >= ACCURACY_GATE:
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
    print(f"PROMOTED {best['name']} v{mv.version} -> production")
else:
    print(f"GATE FAILED: accuracy {best['accuracy']:.4f} < {ACCURACY_GATE}. No promotion.")