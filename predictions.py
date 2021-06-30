import os
import joblib
import pandas as pd


def read_files():
    """Function is loading the data set
    Returns:
        return X_t - data to be predicted
    """
    X = pd.read_csv('test_data.csv')
    return X


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


def saving_to_csv(data: pd.DataFrame, name: str, work_dir: str):
    """Function is saving data to csv file
    Args:
        data: Data to save
        name: (str): Name of the file
        work_dir: working directory
    Returns:
        Creates csv file with given data.
        No other return needed
    """
    return data.to_csv(os.path.join(work_dir, "chart_csv", name))


def predict_lables(x, model):
    """Function is predicting data based on loaded model
    Args:
        x: data to be predicted
        model: model used to predict the data
    """
    y_pred = model.predict(x)
    dirname = os.path.dirname(__file__)
    pd.DataFrame(y_pred).to_csv(dirname + '/plots/predictions.csv')


def main():
    x = read_files()
    model = load_model('SVC1_model.sav')
    predict_lables(x, model)
    pass


if __name__ == '__main__':
    main()