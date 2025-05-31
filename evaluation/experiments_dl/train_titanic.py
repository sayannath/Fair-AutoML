import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the Titanic training dataset from uploaded file
df = pd.read_csv("../../dataset/titanic/train.csv")

# Select useful columns and drop rows with missing values
df = df[
    ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
].dropna()

# Define target and features
y = df["Survived"]
X = df.drop("Survived", axis=1)

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""
Accuracy: 0.8111888111888111

Classification Report:
               precision    recall  f1-score   support

           0       0.78      0.93      0.85        80
           1       0.88      0.67      0.76        63

    accuracy                           0.81       143
   macro avg       0.83      0.80      0.80       143
weighted avg       0.82      0.81      0.81       143
"""
