from functions import existing_dictionary, read_flies_train_test_split, evaluate_model, save_model, model_traning_dummy


def main():
    existing_dictionary("models")
    X_train, X_test, y_train, y_test = read_flies_train_test_split()
    estim = model_traning_dummy(X_train, y_train)
    estim = evaluate_model(estim, y_test, X_test)
    save_model(estim, 'dummy.sav')
    pass


if __name__ == '__main__':
    main()