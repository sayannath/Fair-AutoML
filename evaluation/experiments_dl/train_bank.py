import pandas as pd
from ucimlrepo import fetch_ucirepo  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# Features and targets
X = bank_marketing.data.features
y = bank_marketing.data.targets.squeeze()  # convert from dataframe to Series

# Convert target to binary
y = y.map({"no": 0, "yes": 1})

# One-hot encode categorical columns
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""
Accuracy: 0.8908548048214088

Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.95      0.94      7952
           1       0.55      0.48      0.52      1091

    accuracy                           0.89      9043
   macro avg       0.74      0.72      0.73      9043
weighted avg       0.89      0.89      0.89      9043
"""