import requests
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
rows = data.data[:5].tolist()

payload = {
    "dataframe_split": {
        "columns": list(data.feature_names),
        "data": rows,
    }
}

r = requests.post("http://127.0.0.1:5001/invocations", json=payload)
print("Status:", r.status_code)
print("Response:", r.text)