!pip install imblearn==0.0

!pip install hpsklearn==0.1.0

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier

#Load datasets
def load_data():
    X_train_ = pd.read_csv('train_data.csv', header=None)
    y_labels = pd.read_csv('train_labels.csv', header=None)
    X_test_ = pd.read_csv('test_data.csv', header=None)
    return[X_train_, y_labels, X_test_]
X, y, X_test_data = load_data()

def scale_data(df1, df2):
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df1)
    X_test_scaled = scaler.fit_transform(df2)

    return [X_scaled, X_test_scaled]

X_scaled, X_test_scaled = scale_data(X, X_test_data)

def pca_data(df1: np.array, df2: np.array):
    
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(df1)
    x_test_to_save = pca.fit_transform(df2)
    
    return [X_pca, x_test_to_save]

X_pca, x_test_to_save = pca_data(X_scaled, X_test_scaled)

def data_sampling(df1: np.array, df2: np.array):
    
    over_sampler = RandomOverSampler(sampling_strategy=0.2)

    x_to_save, y_to_save = over_sampler.fit_resample(df1, df2)
    
    return [x_to_save, y_to_save]

x_to_save, y_to_save = data_sampling(X_pca, y)

def save_data(x_train, x_test, y_train):

    np.save('model_x_train.csv', x_train)
    np.save('model_x_test.csv', x_test)
    np.save('model_y_train.csv', y_train)

save_data(x_to_save, x_test_to_save, y_to_save)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=42)

dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train, y_train)
DummyClassifier(strategy='stratified')
y_pred = dummy_clf.predict(X_test)
print(dummy_clf.score(X_test, y_pred))


