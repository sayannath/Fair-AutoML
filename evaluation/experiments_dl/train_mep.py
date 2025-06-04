import os
import sys

import pandas as pd
from aif360.datasets import MEPSDataset19
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier

# Get the directory path containing autosklearn
package_dir = os.path.abspath(os.path.join(os.path.dirname("Fair-AutoML"), "../"))
# Add the directory to sys.path
sys.path.append(package_dir)

from autosklearn.upgrade.metric import (
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    average_odds_difference,
)

# ----------------------------------------------------------------------------
# (a) Identify features vs. label
# ----------------------------------------------------------------------------
LABEL_COL = "Probability"
PROTECTED_COL = "RACE"

# ----------------------------------------------------------------------------
# 1. Load the AIF360 MEPS dataset (as an AIF360 StandardDataset), then split
# ----------------------------------------------------------------------------
#   - DO NOT call .convert_to_dataframe() before splitting, because we want
#     to use AIF360's built-in .split([...]) method on the Dataset object itself.
orig_ds = MEPSDataset19()  # this is an AIF360 StandardDataset
# Split 70% train / 30% test, with shuffling
dataset_orig_train, dataset_orig_test = orig_ds.split([0.7], shuffle=True)

# ----------------------------------------------------------------------------
# 2. Convert each split into a pandas DataFrame, then rename the label column
# ----------------------------------------------------------------------------
# The .convert_to_dataframe() method returns a tuple (df, meta); we only need df.
train_df, _ = dataset_orig_train.convert_to_dataframe()
test_df, _ = dataset_orig_test.convert_to_dataframe()

# Rename 'UTILIZATION' → 'Probability' in each split
train_df = train_df.rename(columns={"UTILIZATION": LABEL_COL})
test_df = test_df.rename(columns={"UTILIZATION": LABEL_COL})

# ----------------------------------------------------------------------------
# 3. Separate features (X), labels (y), and protected attribute (race) for each split
# ----------------------------------------------------------------------------
# 3.1. For training split
X_train_full = train_df.drop(columns=[LABEL_COL])
y_train_full = train_df[LABEL_COL].astype(int).to_numpy()
race_train_full = train_df[PROTECTED_COL].to_numpy()

# 3.2. For test split
X_test_full = test_df.drop(columns=[LABEL_COL])
y_test_full = test_df[LABEL_COL].astype(int).to_numpy()
race_test_full = test_df[PROTECTED_COL].to_numpy()

# ----------------------------------------------------------------------------
# 4. Identify which of the remaining columns (aside from RACE) are categorical vs. numeric
# ----------------------------------------------------------------------------
feats = [c for c in X_train_full.columns if c != PROTECTED_COL]

cat_feats = [
    c
    for c in feats
    if pd.api.types.is_object_dtype(train_df[c])
    or pd.api.types.is_categorical_dtype(train_df[c])
]
num_feats = [c for c in feats if c not in cat_feats]

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# ----------------------------------------------------------------------------
# 6. Fit the pipeline on the training split
# ----------------------------------------------------------------------------
mlp_classifier.fit(X_train_full[feats], y_train_full)

# ----------------------------------------------------------------------------
# 7. Evaluate on the test split (accuracy + ROC‐AUC)
# ----------------------------------------------------------------------------
y_pred_test = mlp_classifier.predict(X_test_full[feats])
y_proba_test = mlp_classifier.predict_proba(X_test_full[feats])[:, 1]

acc_test = accuracy_score(y_test_full, y_pred_test)
roc_test = roc_auc_score(y_test_full, y_proba_test)

print(f"Test Accuracy:  {acc_test:.4f}")
print(f"Test ROC-AUC:   {roc_test:.4f}")
print("\nClassification Report:\n", classification_report(y_test_full, y_pred_test))

print(disparate_impact(dataset_orig_test, y_pred_test, "RACE"))
print(statistical_parity_difference(dataset_orig_test, y_pred_test, "RACE"))
print(equal_opportunity_difference(dataset_orig_test, y_pred_test, y_test_full, "RACE"))
print(average_odds_difference(dataset_orig_test, y_pred_test, y_test_full, "RACE"))

"""
Test Accuracy:  0.8075
Test ROC-AUC:   0.7644

Classification Report:
               precision    recall  f1-score   support

           0       0.90      0.87      0.88      3933
           1       0.45      0.53      0.48       816

    accuracy                           0.81      4749
   macro avg       0.67      0.70      0.68      4749
weighted avg       0.82      0.81      0.81      4749
0.6143313862444636
0.13129651964993871
0.04133028075671763
0.0675724199984044
"""
