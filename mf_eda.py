import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Load datasets
def load_data() -> list:
    X_train_ = pd.read_csv('train_data.csv', header=None)
    y_labels = pd.read_csv('train_labels.csv', header=None)
    X_test_ = pd.read_csv('test_data.csv', header=None)
    return[X_train_, y_labels, X_test_]
X, y, X_test_data = load_data()

#Show dimension of datasets
print('Dimension of training data:', X.shape)
print('Dimension of training labels:', y.shape)
print('Dimension of test data:', X_test_data.shape)

#Show descriptive statistics
print('Descriptive statistics of training data:', X.describe())
print('Descriptive statistics of labels:', y.describe())
print('Descriptive statistics of test data:', X_test_data.describe())

#Function for searching missing values
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

#Show missing values
missing_values_table(X)
missing_values_table(y)
missing_values_table(X_test_data)

#Count skewness of training data and show histogram with this distribution
skew_df = pd.DataFrame(X.skew())
print(skew_df.plot.hist())

#Count values of training labels
print('Values of training labels:', y.value_counts())
