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
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
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
from sklearn.neural_network import MLPClassifier

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

privileged_groups = [{"race": 1}]
unprivileged_groups = [{"race": 0}]

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


class CustomMLPClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(
            self,
            num_units,
            alpha,
            learning_rate_init,
            max_iter,
            tol,
            activation,
            random_state=None,
    ):
        self.num_units = num_units
        self.hidden_layer_sizes = (num_units,)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.activation = activation
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        from sklearn.neural_network import MLPClassifier

        self.estimator = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            tol=self.tol,
            activation=self.activation,
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
            "shortname": "MLP",
            "name": "Multi-Layer Perceptron Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": False,
            "input": [DENSE, SIGNED_DATA, UNSIGNED_DATA],
            "output": [PREDICTIONS],
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        num_units = UniformIntegerHyperparameter(
            "num_units", 50, 500, default_value=100
        )
        alpha = UniformFloatHyperparameter(
            "alpha", 1e-6, 1e-1, log=True, default_value=1e-4
        )
        learning_rate_init = UniformFloatHyperparameter(
            "learning_rate_init", 1e-4, 1.0, log=True, default_value=0.001
        )
        max_iter = UniformIntegerHyperparameter("max_iter", 100, 500, default_value=300)
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-2, log=True, default_value=1e-4
        )
        activation = CategoricalHyperparameter(
            "activation", ["identity", "logistic", "tanh", "relu"], default_value="relu"
        )

        cs.add_hyperparameters(
            [
                num_units,
                alpha,
                learning_rate_init,
                max_iter,
                tol,
                activation,
            ]
        )
        return cs


autosklearn.pipeline.components.classification.add_classifier(CustomMLPClassifier)
cs = CustomMLPClassifier.get_hyperparameter_search_space()
print(cs)


def accuracy(solution, prediction):
    metric_id = 2
    protected_attr = "race"
    with open("test_split.txt") as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = dataset_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        default = MLPClassifier(
            hidden_layer_sizes=(
                100,
            ),  # analogous to n_estimators=200 → 1 hidden layer of 100 units
            alpha=1e-4,  # L2 penalty (default small value)
            learning_rate_init=0.001,  # analogous to learning_rate=0.35
            max_iter=300,  # analogous to max_depth=6 → more iterations
            tol=1e-4,  # convergence tolerance
            activation="relu",  # common default
            random_state=None,
        )
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

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60 * 60,
    memory_limit=10000000,
    include_estimators=["CustomMLPClassifier"],
    ensemble_size=1,
    include_preprocessors=[
        "select_percentile_classification",
        "extra_trees_preproc_for_classification",
        "select_rates_classification",
    ],
    tmp_folder=temp_path,
    delete_tmp_folder_after_terminate=False,
    metric=accuracy_scorer,
)
automl.fit(X_train, y_train)

print(automl.show_models())
cs = automl.get_configuration_space(X_train, y_train)

a_file = open("compas_mlp_spd_race_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_compas_mlp_spd_race_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()

predictions = automl.predict(X_test)
print(predictions)
print(y_test, len(predictions))
print("SPD-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(dataset_orig_test, predictions, "race"))
print(statistical_parity_difference(dataset_orig_test, predictions, "race"))
print(equal_opportunity_difference(dataset_orig_test, predictions, y_test, "race"))
print(average_odds_difference(dataset_orig_test, predictions, y_test, "race"))

from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 score:", f1_score(y_test, predictions))

import json
from utils.file_ops import write_file
from utils.run_history import _get_run_history

write_file(
    "./run_history/compas_mlp_spd_race_run_history.json",
    json.dumps(_get_run_history(automl_model=automl), indent=4),
)
