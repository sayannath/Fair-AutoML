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
    UnParametrizedHyperparameter,
)
import autosklearn.pipeline.components.classification
from autosklearn.Fairea.fairea import create_baseline
from autosklearn.pipeline.components.classification import (
    AutoSklearnClassificationAlgorithm,
)
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE
from autosklearn.util.common import check_none, check_for_bool
import numpy as np
import pandas as pd
from aif360.datasets import MEPSDataset19
from sklearn.ensemble import RandomForestClassifier
import sklearn
import autosklearn.classification
from autosklearn.upgrade.metric import (
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    average_odds_difference,
)

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


class CustomRandomForest(AutoSklearnClassificationAlgorithm):
    def __init__(
            self,
            n_estimators,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            bootstrap,
            max_leaf_nodes,
            min_impurity_decrease,
            max_features="auto",
            max_depth=7,
            random_state=42,
            n_jobs=-1,
            class_weight=None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.estimator = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier

        self.n_estimators = int(self.n_estimators)

        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)

        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

        if self.max_features not in ("sqrt", "log2", "auto"):
            max_features = int(X.shape[1] ** float(self.max_features))
        else:
            max_features = self.max_features

        self.bootstrap = check_for_bool(self.bootstrap)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.min_impurity_decrease = float(self.min_impurity_decrease)

        # initial fit of only increment trees
        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            warm_start=True,
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
            "shortname": "RF",
            "name": "Random Forest Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter(
            "n_estimators", 100, 1000, default_value=542
        )
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini"
        )
        max_features = UniformFloatHyperparameter(
            "max_features", 0.2, 0.9, default_value=0.5689655172413793
        )

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=9
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 2, 20, default_value=9
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True"
        )

        cs.add_hyperparameters(
            [
                n_estimators,
                criterion,
                max_features,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                bootstrap,
                min_impurity_decrease,
            ]
        )
        return cs


# Add custom random forest classifier component to auto-sklearn.
autosklearn.pipeline.components.classification.add_classifier(CustomRandomForest)
cs = CustomRandomForest.get_hyperparameter_search_space()
print(cs)


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
        default = RandomForestClassifier()
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
    beta += 0.2
    if beta > 1.0:
        beta = 1.0
    try:
        num_keys = sum(1 for line in open("num_keys.txt"))
        print(num_keys)
        beta -= 0.050 * int(int(num_keys) / 10)
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


accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu",
    score_func=accuracy,
    optimum=1,
    greater_is_better=False,
    needs_proba=False,
    needs_threshold=False,
)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60 * 60,
    memory_limit=10000000,
    include_estimators=["CustomRandomForest"],
    ensemble_size=1,
    include_preprocessors=[
        "kernel_pca",
        "select_percentile_classification",
        "select_rates_classification",
    ],
    tmp_folder=temp_path,
    delete_tmp_folder_after_terminate=False,
    metric=accuracy_scorer,
)
automl.fit(X_train_full[feats], y_train_full)

###########################################################################
# Get the Score of the final ensemble
# ===================================

print(automl.show_models())
cs = automl.get_configuration_space(X_train_full[feats], y_train_full)

a_file = open("mep_rf_eod_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_mep_rf_eod_60sp" + str(now) + ".pkl", "wb")
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
    "./run_history/mep_rf_eod_race_run_history.json",
    json.dumps(_get_run_history(automl_model=automl), indent=4),
)
