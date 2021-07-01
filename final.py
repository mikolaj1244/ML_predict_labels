import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def read_flies_train_test_split():
    """Function is loading the data and spliting it into X_tarin 90%
     of the tata and Xtest 10 % of the data. y_train contains 90%
     of the test lables and y_test contains 10% of the test lables

    Returns:
        return_1 splits the data into X_train, X_test, y_train, y_test and returns X_t(with data t obe predicted)
    """
    X = pd.read_csv('train_data.csv', header=None)
    y = pd.read_csv('train_labels.csv', header=None)
    X_t = pd.read_csv('test_data.csv', header=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test, X_t


def predict_model(X_train, y_train):
    """Function is fitting the SVC model with train data
    Args:
        X_train: data to be predicted
        y_train: model used to predict the data
    Returns:
        return_1 returns fitted model
    """
    svc = SVC(C=3.397795021344685, cache_size=512, coef0=0, degree=2.0,
        gamma=1.049648344136157, kernel='poly', max_iter=44885307.0, random_state=1,
        shrinking=False, tol=0.0001796410547203976)
    model = svc.fit(X_train, y_train.values.ravel())
    return model


def predict_lables(x, model):
    """Function is predicting data based on loaded model
    Args:
        x: data to be predicted
        model: model used to predict the data
    """
    y_pred = model.predict(x)
    dirname = os.path.dirname(__file__)
    pd.DataFrame(y_pred).to_csv(dirname + '/lables/predictions.csv')


def main():
    X_train, X_test, y_train, y_test, X_t = read_flies_train_test_split()
    model = predict_model(X_train, y_train)
    predict_lables(X_t, model)
    pass


if __name__ == '__main__':
    main()