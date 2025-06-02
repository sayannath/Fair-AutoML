import os
import resource
import sys

# Get the directory path containing autosklearn
package_dir = os.path.abspath(os.path.join(os.path.dirname("Fair-AutoML"), "../.."))
# Add the directory to sys.path
sys.path.append(package_dir)
from xgboost import XGBClassifier

import autosklearn.pipeline.components.classification
import datetime

import pickle

import autosklearn.classification
import autosklearn.metrics
import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np

import sklearn.metrics
import autosklearn.classification
from autosklearn.upgrade.metric import (
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    average_odds_difference,
)
from autosklearn.Fairea.fairea import create_baseline
from ConfigSpace import EqualsCondition, InCondition

from autosklearn.pipeline.implementations.util import softmax
from autosklearn.util.common import check_for_bool, check_none

train_list = "data_orig_train_adult.pkl"
test_list = "data_orig_test_adult.pkl"


def custom_preprocessing(df):
    def group_race(x):
        if x == "White":
            return 1.0
        else:
            return 0.0

    # Recode sex and race
    df["sex"] = df["sex"].replace({"Female": 0.0, "Male": 1.0})
    df["race"] = df["race"].apply(lambda x: group_race(x))
    return df


############################################################################
# File Remover
# ============

now = str(datetime.datetime.now())[:19]
now = now.replace(":", "_")
temp_path = "adult_svm_spd" + str(now)
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
import pandas as pd
from aif360.datasets import StandardDataset

train = pd.read_pickle(train_list)
test = pd.read_pickle(test_list)
na_values = ["?"]
default_mappings = {
    "label_maps": [{1.0: ">50K", 0.0: "<=50K"}],
    "protected_attribute_maps": [
        {1.0: "White", 0.0: "Non-white"},
        {1.0: "Male", 0.0: "Female"},
    ],
}

data_orig_train = StandardDataset(
    df=train,
    label_name="income-per-year",
    favorable_classes=[">50K", ">50K."],
    protected_attribute_names=["sex"],
    privileged_classes=[[1]],
    instance_weights_name=None,
    categorical_features=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "native-country",
    ],
    features_to_keep=[],
    features_to_drop=["income", "native-country", "hours-per-week"],
    na_values=na_values,
    custom_preprocessing=custom_preprocessing,
    metadata=default_mappings,
)
data_orig_test = StandardDataset(
    df=test,
    label_name="income-per-year",
    favorable_classes=[">50K", ">50K."],
    protected_attribute_names=["sex"],
    privileged_classes=[[1]],
    instance_weights_name=None,
    categorical_features=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "native-country",
    ],
    features_to_keep=[],
    features_to_drop=["income", "native-country", "hours-per-week"],
    na_values=na_values,
    custom_preprocessing=custom_preprocessing,
    metadata=default_mappings,
)

privileged_groups = [{"sex": 1}]
unprivileged_groups = [{"sex": 0}]

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
from autosklearn.pipeline.constants import (
    DENSE,
    SPARSE,
    SIGNED_DATA,
    UNSIGNED_DATA,
    PREDICTIONS,
)


# class CustomSVM(AutoSklearnClassificationAlgorithm):
#     def __init__(
#             self,
#             C,
#             kernel,
#             gamma,
#             shrinking,
#             probability=True,
#             random_state=None,
#     ):
#         self.C = C
#         self.kernel = kernel
#         self.gamma = gamma
#         self.shrinking = shrinking
#         self.probability = probability
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         from sklearn.svm import SVC
#
#         self.estimator = SVC(
#             C=self.C,
#             kernel=self.kernel,
#             gamma=self.gamma,
#             shrinking=self.shrinking,
#             probability=self.probability,
#             random_state=self.random_state,
#         )
#         self.estimator.fit(X, y)
#         return self
#
#     def predict(self, X):
#         if self.estimator is None:
#             raise NotImplementedError()
#         return self.estimator.predict(X)
#
#     def predict_proba(self, X):
#         if self.estimator is None:
#             raise NotImplementedError()
#         return self.estimator.predict_proba(X)
#
#     @staticmethod
#     def get_properties(dataset_properties=None):
#         return {
#             "shortname": "SVM",
#             "name": "Support Vector Classifier",
#             "handles_regression": False,
#             "handles_classification": True,
#             "handles_multiclass": True,
#             "handles_multilabel": False,
#             "handles_multioutput": False,
#             "is_deterministic": True,
#             "input": [DENSE, SPARSE, SIGNED_DATA, UNSIGNED_DATA],
#             "output": [PREDICTIONS],
#         }
#
#     @staticmethod
#     def get_hyperparameter_search_space(dataset_properties=None):
#         cs = ConfigurationSpace()
#
#         C = UniformFloatHyperparameter("C", lower=0.03125, upper=32768.0, log=True, default_value=1.0)
#         kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
#         gamma = UniformFloatHyperparameter("gamma", lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
#         shrinking = CategoricalHyperparameter("shrinking", [True, False], default_value=True)
#
#         cs.add_hyperparameters([C, kernel, gamma, shrinking])
#         return cs


class CustomSVC(AutoSklearnClassificationAlgorithm):
    def __init__(
        self,
        C,
        kernel,
        gamma,
        shrinking,
        tol,
        max_iter,
        class_weight=None,
        degree=3,
        coef0=0,
        random_state=42,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.svm

        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM. The cache size is
        # calculated as 2/3 of the available memory (which is calculated as the memory limit minus
        # the used memory)
        try:
            # Retrieve memory limits imposed on the process
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)

            if soft > 0:
                # Convert limit to units of megabytes
                soft /= 1024 * 1024

                # Retrieve memory used by this process
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024

                # In MacOS, the MaxRSS output of resource.getrusage in bytes; on other platforms,
                # it's in kilobytes
                if sys.platform == "darwin":
                    maxrss = maxrss / 1024

                cache_size = (soft - maxrss) / 1.5

                if cache_size < 0:
                    cache_size = 200
            else:
                cache_size = 200
        except Exception:
            cache_size = 200

        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)

        self.shrinking = check_for_bool(self.shrinking)

        if check_none(self.class_weight):
            self.class_weight = None

        self.estimator = sklearn.svm.SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            cache_size=cache_size,
            decision_function_shape="ovr",
        )
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "LibSVM-SVC",
            "name": "LibSVM Support Vector Classification",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    # SVC(C=0.85, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    #     max_iter=-1, random_state=42, shrinking=True, tol=0.001, probability=True,
    #     verbose=False)
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 0.03175, 8186, log=True, default_value=0.85)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(
            name="kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf"
        )
        degree = UniformIntegerHyperparameter("degree", 3, 4, default_value=3)
        gamma = UniformFloatHyperparameter(
            "gamma", 3.3853109172452256e-05, 1.56664, log=True, default_value=0.1
        )
        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter(
            "coef0", -0.12737, 0.67005, default_value=0.0
        )
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter(
            "shrinking", ["True", "False"], default_value="True"
        )
        tol = UniformFloatHyperparameter(
            "tol", 1.0380433896731804e-05, 0.02999, default_value=1e-3, log=True
        )
        # cache size is not a hyperparameter, but an argument to the program!
        max_iter = UnParametrizedHyperparameter("max_iter", -1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [C, kernel, degree, gamma, coef0, shrinking, tol, max_iter]
        )

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs


autosklearn.pipeline.components.classification.add_classifier(CustomSVC)
cs = CustomSVC.get_hyperparameter_search_space()
print(cs)


############################################################################
# Custom metrics definition
# =========================


def accuracy(solution, prediction):
    metric_id = 2
    protected_attr = "sex"
    with open("test_split.txt") as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = data_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        default = XGBClassifier(
            learning_rate=0.35,
            n_estimator=200,
            max_depth=6,
            subsample=1,
            min_child_weight=1,
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
    include_estimators=["CustomSVC"],
    ensemble_size=1,
    # include_preprocessors=[
    #     "select_percentile_classification",
    #     "extra_trees_preproc_for_classification",
    #     "select_rates_classification",
    # ],
    tmp_folder=temp_path,
    delete_tmp_folder_after_terminate=False,
    metric=accuracy_scorer,
)
automl.fit(X_train, y_train)

print(automl.show_models())
cs = automl.get_configuration_space(X_train, y_train)

a_file = open("adult_spd_spd_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_adult_spd_spd_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()

predictions = automl.predict(X_test)
print(predictions)
print(y_test, len(predictions))
print("SPD-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(data_orig_test, predictions, "sex"))
print(statistical_parity_difference(data_orig_test, predictions, "sex"))
print(equal_opportunity_difference(data_orig_test, predictions, y_test, "sex"))
print(average_odds_difference(data_orig_test, predictions, y_test, "sex"))

from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 score:", f1_score(y_test, predictions))

import json
from utils.file_ops import write_file
from utils.run_history import _get_run_history

write_file(
    "./run_history/adult_svm_spd_sex_run_history.json",
    json.dumps(_get_run_history(automl_model=automl), indent=4),
)
