from functions import (evaluate_model, existing_dictionary,
                       model_traning_dummy, read_flies_train_test_split,
                       save_model)


def main():
    existing_dictionary("models")
    X_train, X_test, y_train, y_test = read_flies_train_test_split()
    estim = model_traning_dummy(X_train, y_train)
    estim = evaluate_model(estim, y_test, X_test)
    save_model(estim, 'dummy.sav')
    pass


if __name__ == '__main__':
    main()