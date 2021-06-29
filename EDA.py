import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.ensemble import IsolationForest
import seaborn as sns
import os


def read_files():
    """Function is loading the data set
    Returns:
        return X - train data, X_t - data to be predicted, y - train lables
    """
    X = pd.read_csv('train_data.csv')
    X_t = pd.read_csv('test_data.csv')
    y = pd.read_csv('train_labels.csv')
    return X, X_t, y


def describe(X):
    """Function is ploting describe table
    Args:
        X: train data
    """
    X_short = X.iloc[:, [1, 2, 3, 4, 5]]
    desc = X_short.describe()
    fig = plt.figure(figsize=(10, 10))
    plot = plt.subplot(frame_on=False)
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)
    table(plot, desc,loc='upper right')
    plt.savefig('desc_plot.png')
    dirname = os.path.dirname(__file__)
    fig.savefig(dirname + '/plots/desc_plot.png')


def value_counts(y):
    """Function is ploting value counts table
    Args:
        y: train lables
    """
    df = pd.DataFrame(y.value_counts())
    df_new = df.rename(columns={0: 'value_counts'}, inplace=False)
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(frame_on=False)  # no visible frame
    ax1.xaxis.set_visible(False)  # hide the x axis
    ax1.yaxis.set_visible(False)  # hide the y axis
    table(ax1, df_new)  # where df is your data frame
    dirname = os.path.dirname(__file__)
    fig.savefig(dirname + '/plots/value_counts.png')


def pca_tesne(X,y):
    """Function is using pca + tesne pipeline to plot dataset
    Args:
        X: train data
        y: train lables
    """
    pca_tsne = Pipeline([
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("tsne", TSNE(n_components=2, random_state=42,learning_rate=100, perplexity=50))
    ])
    pca_tsne = pca_tsne.fit_transform(X)
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(x=pca_tsne[:, 0],
                    y=pca_tsne[:, 1],
                    legend=True,
                    hue=y['1'])
    plt.savefig('pca_tesne.png')
    dirname = os.path.dirname(__file__)
    fig.savefig(dirname + '/plots/pca_tesne.png')


def null_detect(X):
    """Function is chceking for null values in the dataset and printing it out.
    Args:
        X: train data
    """
    print(f"null count: {X.isnull().sum().sum()}")



def pca_tesne_ss(X,y):
    """Function is using pca + tesne + StandardScaler pipeline to plot dataset.
    Args:
        X: train data
        y: train lables
    """
    pca_tsne = Pipeline([
    ("ss", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42,learning_rate=100, perplexity=50))
    ])
    pca_tsne = pca_tsne.fit_transform(X)
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(x=pca_tsne[:, 0],
                y=pca_tsne[:, 1],
                legend=True,
                hue=y['1'])
    dirname = os.path.dirname(__file__)
    fig.savefig(dirname + '/plots/pca_tesne_ss.png')


def outliers(X):
    """Function is using IsolationForest algorithm to find outliers it the dataset.
    Args:
        X: train data
    """
    # summarize the shape of the training dataset
    print(f"Data set before outliers detection{X.shape}")
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X = X[mask]
    # summarize the shape of the updated training dataset
    print(f"Data set after outliers detection{X.shape}")


def main():
    X, X_t, y = read_files()
    value_counts(y)
    describe(X)
    pca_tesne(X,y)
    pca_tesne_ss(X,y)
    null_detect(X)
    outliers(X)
    pass


if __name__ == '__main__':
    main()