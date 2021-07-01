%pip install --upgrade --quiet neptune-client

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import neptune.new as neptune
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

#Load datasets
def load_data() -> list:
    X_train_ = pd.read_csv('train_data.csv', header=None)
    y_labels = pd.read_csv('train_labels.csv', header=None)
    X_test_ = pd.read_csv('test_data.csv', header=None)
    return[X_train_, y_labels, X_test_]
X, y, X_test_data = load_data()

#Split training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Run neptune.ai
run = neptune.init(project='ml_cdv/predict-labels',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYWJkNjhjMS04MzYyLTQ2ZDktOGYwZS03NDFhOWM0MjUzYjIifQ==') # your credentials
run

#Function with model, score and confusion matrix
def model(x_1, x_2, y_1, y_2, save: bool = False):

    pipe = Pipeline([('pca', PCA(n_components = 0.95)), ('scaler', MinMaxScaler()),('classifier', SVC())])

    params = [
        {'scaler': [MinMaxScaler()],
        'pca': [PCA(n_components = 0.95)]},
        {"classifier": [LogisticRegression(random_state=42)],
        "classifier__penalty": ["l2"],
        "classifier__C": np.logspace(0.001, 0.1, 10),
        'classifier__class_weight': ['balanced'],
        "classifier__solver": ["liblinear"]
        },

        {'classifier': [SVC(random_state=42)],
        'classifier__kernel': ['linear', 'poly'],
        'classifier__class_weight': ['balanced'],
        'classifier__C': np.logspace(1,2,5)},
        
        {"classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [100, 120, 300, 500, 800, 1200],
        "classifier__max_features": ['log2', 'sqrt', 'auto', None],
        "classifier__max_depth": [5, 8, 15, 25, 30, None],
        "classifier__min_samples_split": [1,2,5,10,15,100],
        "classifier__min_samples_leaf": [1,2,5,10]
        }]

    randsearch = RandomizedSearchCV(pipe,
                              params,
                              cv=2,
                              verbose=1,
                              n_jobs=-1,
                              scoring='f1_micro')
    
    best_model = randsearch.fit(x_2, y_2.values.ravel())
    y_pred = best_model.predict(x_2)
    print(f"\nBest model params: \n{best_model.best_params_}")
    print(f"\nModel scorer: \n{best_model.scorer_}")
    print(f"\nModel score: \n{best_model.best_score_}")
    print(confusion_matrix(y_2, y_pred))

    if save:
        filename = "mf_model.pkl"
        joblib.dump(best_model, filename)

#Run predicting function and save file with model
y_pred = model(X_train, X_test, y_train, y_test, save=True)

#Stop neptune.ai
run.stop()

#Load file with saved model
def load_model(pred):
    
    filename = "mf_model.pkl"
    loaded_model = joblib.load(filename)

    labels = loaded_model.predict(pred)
    
    return labels

#Predict labels with testing data
test_labels = load_model(X_test_data)

#Save dataframe with predicted labels to csv file
test_labels_df = pd.DataFrame(test_labels)
test_labels_df.to_csv("test_labels.csv")

#Read dataframe with predicted labels
test_labels_df

#Count values of dataframe with predicted labels
test_labels_df.value_counts()
