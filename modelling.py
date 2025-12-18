import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load data
X_train = pd.read_csv('maintance_preprocessing/X_train.csv')
y_train = pd.read_csv('maintance_preprocessing/y_train.csv')

mlflow.set_experiment("Predictive_Maintenance_Baseline")
mlflow.xgboost.autolog()

with mlflow.start_run():
    model = XGBClassifier()
    model.fit(X_train, y_train)
    print("Baseline model trained with autolog.")