import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import joblib

df= pd.read_csv("data/scored_data.csv")

percentile = 1

threshold = np.percentile(df["anomaly_score"], percentile)

print("Adaptive Threshold:", threshold)

df["is_anomaly"] = df["anomaly_score"] < threshold

df.to_csv("data/final_output.csv", index=False)
print("Anomaly detection completed")

y_true = df["Class"]
y_pred = df["is_anomaly"].astype(int)
print(classification_report(y_true, y_pred))
joblib.dump(threshold, "src/threshold.pkl")