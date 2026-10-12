import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

BEST_RUN_ID = "4719874f4d644947b9536bb0c91fb7f0"
MODEL_NAME = "BreastCancerClassifier"

# model register karo (version 1 banega)
model_uri = f"runs:/{BEST_RUN_ID}/model"
result = mlflow.register_model(model_uri, MODEL_NAME)

print(f"Registered: {MODEL_NAME} v{result.version}")

# description add karo
client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description="Random Forest on Breast Cancer dataset. Accuracy > 0.96, ROC-AUC > 0.99."
)

# tags add karo
client.set_registered_model_tag(MODEL_NAME, "framework", "sklearn")
client.set_registered_model_tag(MODEL_NAME, "owner", "mlops-lab-student")

# alias lagao (Staging ki jagah)
client.set_registered_model_alias(MODEL_NAME, "production", 1)
print("Alias 'production' set ho gaya!")