"""Sample training script"""

import pickle
from collections import namedtuple
from pathlib import Path
from typing import List

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC


MODELS_DIR = Path("models/")
model_wrappers_list: List[dict] = []


# ================ #
# Data preparation #
# ================ #

# Load data
Iris_data = load_iris()

# Split data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    Iris_data.data, Iris_data.target, test_size=0.3, random_state=4
)


# ================== #
# LogisticRegression #
# ================== #

print("Creating logistic regression model...")

# Define the Logistic Regression model
LR_parameters = {
    "C": 0.1,
    "max_iter": 20,
    "fit_intercept": True,
    "solver": "liblinear",
    "random_state": 0,
}
LR_model = LogisticRegression(**LR_parameters)

# Train the Model
LR_model.fit(Xtrain, Ytrain)

# Evaluate model accuracy by cross-validation
LR_accuracies = cross_val_score(estimator=LR_model, X=Xtrain, y=Ytrain, cv=10)
LR_metrics = {"accuracy": LR_accuracies.mean()}

# Wrap the trained model in a Model namedtuple and append it to the model list
LR_model_dict = {
    "type": "LogisticRegression",
    "params": LR_parameters,
    "model": LR_model,
    "metrics": LR_metrics,
}
model_wrappers_list.append(LR_model_dict)

print("Logistic regression model created.")
print(LR_model_dict, end="\n\n\n")


# =============================== #
# C-Support Vector Classification #
# =============================== #

print("Creating SVC model...")

# Define the SVC model
SVC_parameters = {"kernel": "linear", "random_state": 0}

SVC_model = SVC(**SVC_parameters)

# Train the Model
SVC_model.fit(Xtrain, Ytrain)

# Evaluate model accuracy by cross-validation
SVC_accuracies = cross_val_score(estimator=SVC_model, X=Xtrain, y=Ytrain, cv=10)
SVC_metrics = {"accuracy": SVC_accuracies.mean()}

# Wrap the trained model in a Model namedtuple and append it to the model list
SVC_model_dict = {
    "type": "SVC",
    "params": SVC_parameters,
    "model": SVC_model,
    "metrics": SVC_metrics,
}

model_wrappers_list.append(SVC_model_dict)

print("SVC model created.")
print(SVC_model_dict, end="\n\n\n")


# ============= #
# Serialization #
# ============= #

print("Serializing model wrappers...")

for wrapped_model in model_wrappers_list:

    pkl_filename = f"{wrapped_model['type']}_model.pkl"
    pkl_path = MODELS_DIR / pkl_filename

    with open(pkl_path, "wb") as file:
        pickle.dump(wrapped_model, file)

print("Serializing completed.")