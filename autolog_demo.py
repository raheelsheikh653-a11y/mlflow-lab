import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV

# Sirf ek line se sab kuch auto log hoga
mlflow.sklearn.autolog()

mlflow.set_experiment("autolog-demo")

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=0
)

# GridSearchCV har combination try karta hai
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}

gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1
)

with mlflow.start_run(run_name="gridsearch-autolog"):
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    print("Best CV score:", round(gs.best_score_, 4))