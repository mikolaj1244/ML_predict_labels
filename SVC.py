from functions import existing_dictionary, read_flies_train_test_split, model_traning, evaluate_model, save_model
from hpsklearn import svc


def main():
    existing_dictionary("models")
    X_train, X_test, y_train, y_test = read_flies_train_test_split()
    estim = model_traning(X_train, y_train, svc('mySVC'), 500)
    estim = evaluate_model(estim, y_test, X_test)
    save_model(estim, "SVC1_model.sav")
    pass


if __name__ == '__main__':
    main()