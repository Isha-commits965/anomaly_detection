import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

df= pd.read_csv("data/creditcard.csv")
X= df.drop("Class", axis=1)
model = IsolationForest(
    n_estimators=100,
    contamination=0.002,
    random_state=42
)
model.fit(X)

scores = model.decision_function(X)
df["anomaly_score"]= scores
df.to_csv("data/scored_data.csv", index= False)
joblib.dump(model, "src/model.pkl")
