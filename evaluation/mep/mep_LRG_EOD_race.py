import datetime
import os
import pickle
import sys


# Get the directory path containing autosklearn
package_dir = os.path.abspath(os.path.join(os.path.dirname("Fair-AutoML"), "../.."))
# Add the directory to sys.path
sys.path.append(package_dir)

import autosklearn.pipeline.components.classification
from autosklearn.Fairea.fairea import create_baseline
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SIGNED_DATA
import pandas as pd
from aif360.datasets import MEPSDataset19
import sklearn
import autosklearn.classification
from autosklearn.upgrade.metric import (
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    average_odds_difference,
)
from sklearn.linear_model import LogisticRegression

now = str(datetime.datetime.now())[:19]
now = now.replace(":", "_")
temp_path = "mep_LRG_EOD_race" + str(now)
try:
    os.remove("test_split.txt")
except:
    pass
try:
    os.remove("num_keys.txt")
except:
    pass
try:
    os.remove("beta.txt")
except:
    pass

f = open("beta.txt", "w")
f.close()

LABEL_COL = "Probability"
PROTECTED_COL = "RACE"

orig_ds = MEPSDataset19()  # this is an AIF360 StandardDataset
dataset_orig_train, dataset_orig_test = orig_ds.split([0.7], shuffle=True)

train_df, _ = dataset_orig_train.convert_to_dataframe()
test_df, _ = dataset_orig_test.convert_to_dataframe()

# Rename 'UTILIZATION' → 'Probability' in each split
train_df = train_df.rename(columns={"UTILIZATION": LABEL_COL})
test_df = test_df.rename(columns={"UTILIZATION": LABEL_COL})

X_train_full = train_df.drop(columns=[LABEL_COL])
y_train_full = train_df[LABEL_COL].astype(int).to_numpy()
race_train_full = train_df[PROTECTED_COL].to_numpy()

X_test_full = test_df.drop(columns=[LABEL_COL])
y_test_full = test_df[LABEL_COL].astype(int).to_numpy()
race_test_full = test_df[PROTECTED_COL].to_numpy()

feats = [c for c in X_train_full.columns if c != PROTECTED_COL]
cat_feats = [
    c
    for c in feats
    if pd.api.types.is_object_dtype(train_df[c])
       or pd.api.types.is_categorical_dtype(train_df[c])
]
num_feats = [c for c in feats if c not in cat_feats]

print(X_train_full[feats].shape)
print(y_train_full.shape)
print(X_test_full[feats].shape)
print(y_test_full.shape)

privileged_groups = [{"RACE": 1}]
unprivileged_groups = [{"RACE": 0}]

import numpy as np
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)


class CustomLRG(AutoSklearnClassificationAlgorithm):
    def __init__(self, penalty, C, dual, random_state=None):
        self.penalty = penalty
        self.C = C
        self.dual = dual
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        self.estimator = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            dual=self.dual,
            random_state=self.random_state,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "LRG",
            "name": "LRG Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": False,
            # Both input and output must be tuple(iterable)
            "input": [DENSE, SIGNED_DATA, UNSIGNED_DATA],
            "output": [PREDICTIONS],
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.

        penalty = CategoricalHyperparameter(
            name="penalty", choices=["l2"], default_value="l2"
        )
        C = CategoricalHyperparameter(
            name="C",
            choices=[1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0],
            default_value=1.0,
        )
        dual = CategoricalHyperparameter(
            name="dual", choices=[False], default_value=False
        )

        cs.add_hyperparameters([penalty, C, dual])
        return cs


autosklearn.pipeline.components.classification.add_classifier(CustomLRG)
cs = CustomLRG.get_hyperparameter_search_space()
print(cs)


############################################################################
# Custom metrics definition
# =========================
def accuracy(solution, prediction):
    metric_id = 3
    protected_attr = "RACE"
    with open("test_split.txt") as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = dataset_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        default = LogisticRegression()
        degrees = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        mutation_strategies = {"0": [1, 0], "1": [0, 1]}
        dataset_orig = subset_data_orig_train
        res = create_baseline(
            default,
            dataset_orig,
            privileged_groups,
            unprivileged_groups,
            data_splits=10,
            repetitions=10,
            odds=mutation_strategies,
            options=[0, 1],
            degrees=degrees,
        )
        acc0 = np.array(
            [np.mean([row[0] for row in res["0"][degree]]) for degree in degrees]
        )
        acc1 = np.array(
            [np.mean([row[0] for row in res["1"][degree]]) for degree in degrees]
        )
        fair0 = np.array(
            [
                np.mean([row[metric_id] for row in res["0"][degree]])
                for degree in degrees
            ]
        )
        fair1 = np.array(
            [
                np.mean([row[metric_id] for row in res["1"][degree]])
                for degree in degrees
            ]
        )

        if min(acc0) > min(acc1):
            beta = (max(acc0) - min(acc0)) / (max(acc0) - min(acc0) + max(fair0))
        else:
            beta = (max(acc1) - min(acc1)) / (max(acc1) - min(acc1) + max(fair1))

        f = open("beta.txt", "w")
        f.write(str(beta))
        f.close()
    else:
        f = open("beta.txt", "r")
        beta = float(f.read())
        f.close()
        # print('yyyy')
    # print(beta)
    beta += 0.2
    if beta > 1.0:
        beta = 1.0

    try:
        num_keys = sum(1 for line in open("num_keys.txt"))
        print(num_keys)
        beta -= 0.050 * int(int(num_keys) / 10)
        if beta < 0.0:
            beta = 0
        if int(num_keys) % 10 == 0:
            os.remove(temp_path + "/.auto-sklearn/ensemble_read_losses.pkl")
        f.close()
    except FileNotFoundError:
        pass
    fairness_metrics = [
        1 - np.mean(solution == prediction),
        disparate_impact(subset_data_orig_train, prediction, protected_attr),
        statistical_parity_difference(
            subset_data_orig_train, prediction, protected_attr
        ),
        equal_opportunity_difference(
            subset_data_orig_train, prediction, solution, protected_attr
        ),
        average_odds_difference(
            subset_data_orig_train, prediction, solution, protected_attr
        ),
    ]

    # print(
    #     fairness_metrics[metric_id],
    #     1 - np.mean(solution == prediction),
    #     fairness_metrics[metric_id] * beta
    #     + (1 - np.mean(solution == prediction)) * (1 - beta),
    #     beta,
    # )

    return fairness_metrics[metric_id] * beta + (
            1 - np.mean(solution == prediction)
    ) * (1 - beta)


############################################################################
# Second example: Use own accuracy metric
# =======================================
print("#" * 80)
print("Use self defined accuracy metric")
accuracy_scorer = autosklearn.metrics.make_scorer(
    name="fair+acc",
    score_func=accuracy,
    optimum=1,
    greater_is_better=False,
    needs_proba=False,
    needs_threshold=False,
)

############################################################################
# Build and fit a classifier
# ==========================
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60 * 60,
    memory_limit=10000000,
    include_estimators=["CustomLRG"],
    ensemble_size=1,
    tmp_folder=temp_path,
    delete_tmp_folder_after_terminate=False,
    metric=accuracy_scorer,
)
automl.fit(X_train_full[feats], y_train_full)

print(automl.show_models())
cs = automl.get_configuration_space(X_train_full[feats], y_train_full)

a_file = open("adult_lrg_eod_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_adult_lrg_eod_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()

predictions = automl.predict(X_test_full[feats])
print(predictions)
print(y_test_full, len(predictions))
print("EOD-Accuracy score:", sklearn.metrics.accuracy_score(y_test_full, predictions))
print(disparate_impact(dataset_orig_test, predictions, "RACE"))
print(statistical_parity_difference(dataset_orig_test, predictions, "RACE"))
print(equal_opportunity_difference(dataset_orig_test, predictions, y_test_full, "RACE"))
print(average_odds_difference(dataset_orig_test, predictions, y_test_full, "RACE"))

from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision:", precision_score(y_test_full, predictions))
print("Recall:", recall_score(y_test_full, predictions))
print("F1 score:", f1_score(y_test_full, predictions))

import json
from utils.file_ops import write_file
from utils.run_history import _get_run_history

write_file(
    "./run_history/mep_lrg_eod_race_run_history.json",
    json.dumps(_get_run_history(automl_model=automl), indent=4),
)
