import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
import sqlite3
import numpy as np
from sklearn.calibration import calibration_curve

# Database connection
DATABASE_PATH = "greyhound_racing_data.db"

# Load data
conn = sqlite3.connect(DATABASE_PATH)
query = """
SELECT rd.dog_id, rd.dog_name, rd.box_number, rd.finish_position, rd.weight,
       rd.win_probability, rd.place_probability, rm.race_date
FROM dog_race_data rd
JOIN race_metadata rm ON rd.race_id = rm.race_id
WHERE rm.race_date IS NOT NULL
"""
data = pd.read_sql(query, conn)
conn.close()

data['race_date'] = pd.to_datetime(data['race_date'], format='%d %B %Y', errors='coerce')
data.sort_values('race_date', inplace=True)

data['target'] = (data['finish_position'] == 1).astype(int)

features = [
    "box_number",
    "weight",
    "win_probability",
    "place_probability"
]

# Rolling-window split
train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
train_data, temp_data = train_test_split(data, train_size=train_size, shuffle=False)
val_data, test_data = train_test_split(temp_data, train_size=val_size, shuffle=False)

# Handle missing values for all sets
train_data = train_data.dropna(subset=features)
val_data = val_data.dropna(subset=features)
test_data = test_data.dropna(subset=features)

# Prepare data
X_train, y_train = train_data[features], train_data['target']
X_val, y_val = val_data[features], val_data['target']
X_test, y_test = test_data[features], test_data['target']

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100)
gb_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return accuracy, logloss, auc, brier

logistic_scores = evaluate_model(logistic_model, X_test, y_test)
gb_scores = evaluate_model(gb_model, X_test, y_test)

# Print results
def print_scores(name, scores):
    print(f"{name} Scores:")
    print(f"Accuracy: {scores[0]:.4f}")
    print(f"Log Loss: {scores[1]:.4f}")
    print(f"AUC: {scores[2]:.4f}")
    print(f"Brier Score: {scores[3]:.4f}")

print_scores("Logistic Regression", logistic_scores)
print_scores("Gradient Boosting", gb_scores)

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, gb_model.predict_proba(X_test)[:, 1], n_bins=10)
np.savetxt("calibration_plan.csv", np.column_stack((prob_true, prob_pred)), delimiter=",", header="True,Predicted", comments="")
