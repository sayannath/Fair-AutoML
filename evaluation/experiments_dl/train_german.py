from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
german = fetch_ucirepo(id=144)  # German Credit Data

# Data and labels
X = german.data.features
y = german.data.targets.squeeze()

# Binary encode the target: 1 (good) => 1, 2 (bad) => 0
y = y.map({1: 1, 2: 0})

# One-hot encode categorical features
X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""
Accuracy: 0.795

Classification Report:
               precision    recall  f1-score   support

           0       0.68      0.58      0.62        59
           1       0.83      0.89      0.86       141

    accuracy                           0.80       200
   macro avg       0.76      0.73      0.74       200
weighted avg       0.79      0.80      0.79       200
"""
