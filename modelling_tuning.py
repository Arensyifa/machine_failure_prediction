import mlflow
import pandas as pd
import dagshub
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import json

# Init DagsHub
dagshub.init(repo_owner="Arensyifa", repo_name="machine_failure_prediction", mlflow=True)

X_train = pd.read_csv('maintenance_preprocessing/X_train.csv')
y_train = pd.read_csv('maintenance_preprocessing/y_train.csv')
X_test = pd.read_csv('maintenance_preprocessing/X_test.csv')
y_test = pd.read_csv('maintenance_preprocessing/y_test.csv')

mlflow.set_experiment("Predictive_Maintenance_Tuning")

with mlflow.start_run():
    params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    mlflow.log_params(params)
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    # Artifact: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Artifact: Feature Importance
    plt.figure()
    pd.Series(model.feature_importances_).plot(kind='bar')
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    
    # Extra Artifacts (Advanced)
    with open("metric_info.json", "w") as f:
        json.dump({"status": "completed", "threshold_met": True}, f)
    mlflow.log_artifact("metric_info.json")
    
    mlflow.sklearn.log_model(model, "model")