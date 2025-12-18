import requests
import json

data = {"dataframe_split": {"columns": ["Type", "Air temp", "Process temp", "RPM", "Torque", "Tool wear"], "data": [[1, 298.1, 308.6, 1551, 42.8, 0]]}}
response = requests.post("http://localhost:5000/invocations", json=data)
print(f"Prediction: {response.json()}")