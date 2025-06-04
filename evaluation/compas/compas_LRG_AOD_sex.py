import datetime
import os
import pickle
import sys

# Get the directory path containing autosklearn
package_dir = os.path.abspath(os.path.join(os.path.dirname("Fair-AutoML"), "../.."))
# Add the directory to sys.path
sys.path.append(package_dir)

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)
import autosklearn.pipeline.components.classification
from autosklearn.Fairea.fairea import create_baseline
from autosklearn.pipeline.components.classification import (
    AutoSklearnClassificationAlgorithm,
)
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SIGNED_DATA
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
import sklearn
import autosklearn.classification
from autosklearn.upgrade.metric import (
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    average_odds_difference,
)
from sklearn.linear_model import LogisticRegression

############################################################################
# File Remover
# ============
now = str(datetime.datetime.now())[:19]
now = now.replace(":", "_")
temp_path = "temp" + str(now)
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
############################################################################
# Data Loading
# ============
df = pd.read_csv("../../dataset/compas/compas-scores-two-years.csv")
print("Dataset shape: ", df.shape)

default_mappings = {
    "label_maps": [{1.0: "Did recid.", 0.0: "No recid."}],
    "protected_attribute_maps": [
        {0.0: "Male", 1.0: "Female"},
        {1.0: "Caucasian", 0.0: "Not Caucasian"},
    ],
}


def default_preprocessing(_df):
    return _df[
        (_df.days_b_screening_arrest <= 30)
        & (_df.days_b_screening_arrest >= -30)
        & (_df.is_recid != -1)
        & (_df.c_charge_degree != "O")
        & (_df.score_text != "N/A")
        ]


dataset_orig = StandardDataset(
    df=df,
    label_name="two_year_recid",
    favorable_classes=[0],
    protected_attribute_names=["sex", "race"],
    privileged_classes=[["Female"], ["Caucasian"]],
    instance_weights_name=None,
    categorical_features=["age_cat", "c_charge_degree", "c_charge_desc"],
    features_to_keep=[
        "sex",
        "age",
        "age_cat",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "c_charge_desc",
        "two_year_recid",
    ],
    features_to_drop=[],
    na_values=[],
    custom_preprocessing=default_preprocessing,
    metadata=default_mappings,
)

privileged_groups = [{"sex": 1}]
unprivileged_groups = [{"sex": 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
dataset_orig_train, dataset_orig_val = dataset_orig_train.split([0.8], shuffle=True)

X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()

X_val = dataset_orig_val.features
y_val = dataset_orig_val.labels.ravel()

X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()

print("Train Shape: ", X_train.shape)
print("Train Label Shape: ", y_train.shape)

print("Validation Shape: ", X_val.shape)
print("Validation Label Shape: ", y_val.shape)

print("Test Shape: ", X_test.shape)
print("Test Label Shape: ", y_test.shape)


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
    metric_id = 4
    protected_attr = "sex"
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

    print(
        fairness_metrics[metric_id],
        1 - np.mean(solution == prediction),
        fairness_metrics[metric_id] * beta
        + (1 - np.mean(solution == prediction)) * (1 - beta),
        beta,
    )

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
automl.fit(X_train, y_train)

###########################################################################
# Get the Score of the final ensemble
# ===================================

print(automl.show_models())
cs = automl.get_configuration_space(X_train, y_train)

a_file = open("compas_lrg_aod_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_compas_lrg_aod_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()

predictions = automl.predict(X_test)
print(predictions)
print(y_test, len(predictions))
print("AOD-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(dataset_orig_test, predictions, "sex"))
print(statistical_parity_difference(dataset_orig_test, predictions, "sex"))
print(equal_opportunity_difference(dataset_orig_test, predictions, y_test, "sex"))
print(average_odds_difference(dataset_orig_test, predictions, y_test, "sex"))

from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 score:", f1_score(y_test, predictions))

import json
from utils.file_ops import write_file
from utils.run_history import _get_run_history

write_file(
    "./run_history/compas_lrg_aod_sex_run_history.json",
    json.dumps(_get_run_history(automl_model=automl), indent=4),
)
