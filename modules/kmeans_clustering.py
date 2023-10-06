import gensim
from sklearn.cluster import KMeans
from sklearn import preprocessing
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import normalize

def wordcloud_clusters(model, vectors, features, n_top_words=50):
    num_clouds = len(np.unique(model.labels_))
    fig, axs = plt.subplots(num_clouds, 1, figsize=(6.4, 6.4 * num_clouds))
    axs = axs.flatten()

    for i, cluster in enumerate(np.unique(model.labels_)):
        ax = axs[i]
        size = {}
        words = vectors[model.labels_ == cluster].sum(axis=0)
        largest = words.argsort()[::-1]
        for j in range(0, n_top_words):
            size[features[largest[j]]] = abs(words[largest[j]])
        wc = WordCloud(
            background_color="white",
            max_words=100,
            width=500,
            height=300
        )
        wc.generate_from_frequencies(size)
        # plt.figure(figsize=(4, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f'Cluster = {cluster}')

    fig.tight_layout()


def fit_kmeans(corpus, dictionary, K=3, word_cloud=True, to_array=False):
    fomc_norm = preprocessing.normalize(corpus)
    if to_array:
        fomc_norm = fomc_norm.toarray()

    k_means_fomc = KMeans(n_clusters=K, random_state=42)
    k_means_fomc.fit(fomc_norm)
    sizes_df = pd.DataFrame.from_dict({
        'CLUSTER': [i for i in range(K)],
        'CLUSTER_SIZE': [np.sum(k_means_fomc.labels_ == i) for i in range(K)],
    }).set_index('CLUSTER')
    print(sizes_df)
    if word_cloud:
        wordcloud_clusters(
            k_means_fomc,
            fomc_norm,
            dictionary
        )
    return k_means_fomc

def predict_kmeans(clu, corpus):
    cluster_labels = clu.predict(normalize(corpus))
    return cluster_labels


def generate_labels(df, date_col, label_col, start_date=datetime(1985, 1, 1), end_date=datetime(2023, 6, 14)):
    full_date_range = pd.date_range(
        start=start_date, end=end_date)
    label_data = pd.DataFrame().from_dict({date_col: full_date_range})
    label_data = pd.merge(label_data, df[[
                          date_col, label_col]], on=date_col, how="left").fillna(method="bfill")
    label_data[label_col] = label_data[label_col].astype("int")
    # label_data.to_csv("./assets/" + label_col + ".csv")
    return label_data
