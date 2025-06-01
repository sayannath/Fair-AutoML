from sklearn.datasets import fetch_openml
import pandas as pd

# Load from OpenML
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame

# Drop rows with missing values (marked with '?')
df = df[~df.isin(["?"]).any(axis=1)]

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Encode the labels to 0/1
y = y.map({"<=50K": 0, ">50K": 1})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# from sklearn.neural_network import MLPClassifier
#
# clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
# clf.fit(X_train, y_train)
#
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#
# y_pred = clf.predict(X_test)
#
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# """
# Accuracy: 0.8363189681646023
#
# Classification Report:
#                precision    recall  f1-score   support
#
#            0       0.88      0.90      0.89      7479
#            1       0.66      0.61      0.64      2290
#
#     accuracy                           0.84      9769
#    macro avg       0.77      0.76      0.77      9769
# weighted avg       0.83      0.84      0.83      9769
# """

# SVM classifier
from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
