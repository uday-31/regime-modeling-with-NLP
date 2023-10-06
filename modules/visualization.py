import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud


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
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f'Cluster = {cluster}')
    
    fig.tight_layout()
