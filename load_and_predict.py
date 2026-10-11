import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer

MODEL_NAME = "BreastCancerClassifier"

# alias se model load karo
model_uri = f"models:/{MODEL_NAME}@production"

# Option A: pyfunc (framework-agnostic)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

data = load_breast_cancer()
sample = pd.DataFrame(data.data[:5], columns=data.feature_names)

preds = pyfunc_model.predict(sample)
print("pyfunc predictions:", preds)

# Option B: sklearn native (predict_proba milta hai)
sk_model = mlflow.sklearn.load_model(model_uri)
proba = sk_model.predict_proba(sample)[:, 1]
print("probability:", proba.round(3))