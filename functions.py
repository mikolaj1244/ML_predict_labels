from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from hpsklearn import HyperoptEstimator
import joblib
import pandas as pd
from os import path
import os


def existing_dictionary(file: str):
    """Function is checking if dictionary is
     already existing, and creating it
    if is not.
    Args:
        file: Name of the dictionary that is being checked
    Returns:
        return_1 creates dictionary with given name (file)
        return_2 returns None if folders are already existing.
    """
    if not path.exists(file):
        return os.mkdir(file)
    return None


def read_flies_train_test_split():
    """Function is loading the data and spliting it into X_tarin 90%
     of the tata and Xtest 10 % of the data. y_train contains 90%
     of the test lables and y_test contains 10% of the test lables

    Returns:
        return_1 splits the data into X_train, X_test, y_train, y_test
    """
    X = pd.read_csv('train_data.csv')
    X_t = pd.read_csv('test_data.csv')
    y = pd.read_csv('train_labels.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


def model_traning(X_train, y_train, classifier, max_evals=int):
    """Function trains the model using hpsklearn
    Args:
        X_train: train data
        y_train: train lables
        classifier: choose classifier
        max_evals: number of itteracions for HyperoptEstimator to perform d
    Returns:
        return returns model trained on the train data.
    """
    estim = HyperoptEstimator(classifier = classifier, max_evals=max_evals, trial_timeout=60, seed=42)
    estim.fit(X_train, y_train['1'])
    return estim


def model_traning_dummy(X_train, y_train):
    """Function trains dummy clasifier
    Args:
        X_train: train data
        y_train: train lables
    Returns:
        return returns model trained on the train data.
    """
    estim = DummyClassifier(strategy='uniform')
    estim.fit(X_train, y_train['1'])
    return estim


def evaluate_model(estim, y_test, X_test):
    """Function prints confusion matrix.
    Args:
        y_test: train data
        X_test: train lables
    Returns:
        return returns model trained on the train data.
    """
    y_pred = estim.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    return estim

def save_model(estim, name):
    """Function saves model using joblib
    Args:
        estim: model
        name: name of the file
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f"models/{name}")
    joblib.dump(estim, filename)