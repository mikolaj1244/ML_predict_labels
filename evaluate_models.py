from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd
import os
from functions import  read_flies_train_test_split
import matplotlib.pyplot as plt
import seaborn as sn


def evaluate_model(model, y_test, X_test):
    """Function is evaluating model with f0.5 metric
    Args:
        model: model loaded using joblib
        y_test: 10% of the lables
        X_test: 10% of the data
    Returns:
        return score of a model based on f0.5 metric
    """
    y_pred = model.predict(X_test)
    score = fbeta_score(y_test, y_pred, average='macro', beta=0.5)
    return score


def cm(model, y_test, X_test, names):
    """Function is creating fig with confiusion matrix and saves it to dictionary confiusion_maps
    Args:
        model: model loaded using joblib
        y_test: 10% of the lables
        X_test:0% of the data
        names: name of evaluation model

    Returns:
        return score of a model based on f0.5 metric
    """
    y_pred = model.predict(X_test)
    cm = (confusion_matrix(y_test, y_pred))
    print(cm)
    df_cm = pd.DataFrame(cm)
    fig = plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 16})  # font size
    plt.title(f'confiusion matrix for {names}', fontsize=16, color='#323232')
    plt.show()
    dirname = os.path.dirname(__file__)
    fig.savefig(dirname + f'/confiusion_maps/{names}.png')


def get_cm(svc, dummy, knn, extra_trees, y_test, X_test, names):
    """Function is coaling cm function multiple times to genetate confiusion maps for each model
    Args:
        svc: svc model
        dummy: dummy classifier model
        knn: knn model
        extra_trees: extra trees model
        y_test: 10% of the lables
        X_test:0% of the data
        names: name of model

    Returns:
        return score of a model based on f0.5 metric
    """
    cm(svc,y_test, X_test, names[0])
    cm(dummy, y_test, X_test, names[1])
    cm(knn, y_test, X_test, names[2])
    cm(extra_trees, y_test, X_test, names[3])


def load_model(name: str):
    """Function is loading models form models dictionary
    Args:
        name: name of model
    Returns:
        return returns loaded model
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f"models/{name}")
    model = joblib.load(filename, mmap_mode=None)
    return model


def load_all_models():
    """Function calling load_models and loading models to diffrent variables
    Returns:
        return returns loaded models
    """
    svc = load_model('SVC1_model.sav')
    dummy = load_model('dummy.sav')
    knn = load_model('knn_model.sav')
    extra_trees = load_model('extra_trees_model.sav')
    return svc, dummy, knn, extra_trees


def get_scores(svc, dummy, knn, extra_trees, y_test, X_test):
    """Function is coaling evaluate_model function multiple times to append scores to scores list.
    Args:
        svc: svc model
        dummy: dummy classifier model
        knn: knn model
        extra_trees: extra trees model
        y_test: 10% of the lables
        X_test: 0% of the data

    Returns:
        return1: score of a model based on f0.5 metric
        return2: names list of model names
    """
    scores = []
    names = ["svc", "dummy", "knn", "extra trees"]
    scores.append(evaluate_model(svc, y_test, X_test))
    scores.append(evaluate_model(dummy, y_test, X_test))
    scores.append(evaluate_model(knn, y_test, X_test))
    scores.append(evaluate_model(extra_trees, y_test, X_test))
    print(scores)
    print(names)
    return scores, names


def plot(scores, names):
    """Function is ploting score list and names list which contains sores and names of models.
    Args:
        scores: list with scores of models
        names: names of models
    Returns:
        return1: score of a model based on f0.5 metric
        return2: names list of model names
    """
    plt.figure(figsize=(10, 5))
    plt.bar(names, scores, color='#969696')
    plt.xlabel('models', fontsize=12, color='#323232')
    plt.ylabel('f0.5score', fontsize=12, color='#323232')
    plt.title('scores of diffrent models', fontsize=16, color='#323232')
    plt.show()


def main():
    svc, dummy, knn, extra_trees = load_all_models()
    X_train, X_test, y_train, y_test = read_flies_train_test_split()
    scores, names = get_scores(svc, dummy, knn, extra_trees, y_test, X_test)
    get_cm(svc, dummy, knn, extra_trees, y_test, X_test, names)
    plot(scores, names)
    pass


if __name__ == '__main__':
    main()