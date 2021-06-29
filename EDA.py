import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sns
import os


def read_files():
    X = pd.read_csv('train_data.csv')
    X_t = pd.read_csv('test_data.csv')
    y = pd.read_csv('train_labels.csv')
    return X, X_t, y


def describe(X):
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


def pca_tesne_ss(X,y):
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


def main():
    X, X_t, y = read_files()
    value_counts(y)
    describe(X)
    pca_tesne(X,y)
    pca_tesne_ss(X,y)
    pass


if __name__ == '__main__':
    main()