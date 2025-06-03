import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
df = pd.read_csv(url)

# Filter to only the needed rows and features
df = df[
    (df["days_b_screening_arrest"] <= 30)
    & (df["days_b_screening_arrest"] >= -30)
    & (df["is_recid"] != -1)
    & (df["c_charge_degree"] != "O")
    & (df["score_text"] != "N/A")
    ]

# Target variable: whether someone reoffended within 2 years
y = df["two_year_recid"]

# Selected features
# Instead of a set, use a list of column names:
features = [
    "age",
    "sex",
    "race",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
]
X = df[features]

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

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

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""
Accuracy: 0.6720647773279352

Classification Report:
               precision    recall  f1-score   support

           0       0.68      0.77      0.72       683
           1       0.66      0.55      0.60       552

    accuracy                           0.67      1235
   macro avg       0.67      0.66      0.66      1235
weighted avg       0.67      0.67      0.67      1235
"""
