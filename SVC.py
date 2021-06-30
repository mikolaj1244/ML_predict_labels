from hpsklearn import svc

from functions import (evaluate_model, existing_dictionary, model_traning,
                       read_flies_train_test_split, save_model)


def main():
    existing_dictionary("models")
    X_train, X_test, y_train, y_test = read_flies_train_test_split()
    estim = model_traning(X_train, y_train, svc('mySVC'), 500)
    estim = evaluate_model(estim, y_test, X_test)
    save_model(estim, "SVC1_model.sav")
    pass


if __name__ == '__main__':
    main()