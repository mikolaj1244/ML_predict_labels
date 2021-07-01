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
from imblearn.metrics import classification_report_imbalanced

#Load datasets
def load_data():
    X_train_ = pd.read_csv('train_data.csv', header=None)
    y_labels = pd.read_csv('train_labels.csv', header=None)
    X_test_ = pd.read_csv('test_data.csv', header=None)
    return[X_train_, y_labels, X_test_]
X, y, X_test_data = load_data()

#Function with MinMaxScaler for feature scaling
def scale_data(df1, df2):
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df1)
    X_test_scaled = scaler.fit_transform(df2)

    return [X_scaled, X_test_scaled]

#Run scaling features function
X_scaled, X_test_scaled = scale_data(X, X_test_data)

#Function with PCA
def pca_data(df1, df2):
    
    print('Dimension of training data before PCA:', X_scaled.shape)

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(df1)
    x_test_to_save = pca.fit_transform(df2)

    print('Dimension of training data after PCA:', X_pca.shape)
    
    return [X_pca, x_test_to_save]

#Run PCA and check dimension of data
X_pca, x_test_to_save = pca_data(X_scaled, X_test_scaled)

#Function for oversampling training data
def data_sampling(df1, df2):

    print('Dimension of training data before sampling:', X_pca.shape)
    print('Dimension of training labels before sampling:', y.shape)
    
    over_sampler = RandomOverSampler(sampling_strategy=0.2)

    x_to_save, y_to_save = over_sampler.fit_resample(df1, df2)

    print('Dimension of training data after sampling:', x_to_save.shape)
    print('Dimension of training labels after sampling:', X_pca.shape)
    
    return [x_to_save, y_to_save]

#Run oversampling and check dimension of data
x_to_save, y_to_save = data_sampling(X_pca, y)

#Function for saving numpy arrays
def save_data(x_train, x_test, y_train):

    np.save('model_x_train', x_train)
    np.save('model_x_test', x_test)
    np.save('model_y_train', y_train)

#Save preprocessed data
save_data(x_to_save, x_test_to_save, y_to_save)

#Split training data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=42)

#Fit-predict with dummy classifier and print results with classification report
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train, y_train)
DummyClassifier(strategy='stratified')
y_pred = dummy_clf.predict(X_test)
print(classification_report_imbalanced(y_test, y_pred))
