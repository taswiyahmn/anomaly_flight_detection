import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class BirchClusteringApp:
    def __init__(self, df, kmax=10):
        self.df = df
        self.kmax = kmax
        self.x = df[['avg_elevation', 'avg_diff_vs']]
        self.sil_scores = []
        self.db_scores = []
        self.ch_scores = []

    def run_clustering(self):
        st.title('Clustering Scores with Birch')
        if self.kmax is None:
            st.write("Please provide a value for kmax.")
            return

        st.write(f"Running Birch clustering with kmax={self.kmax}...")

        fig, axes = plt.subplots(nrows=1, ncols=self.kmax-1, figsize=(25, 5))
        ks = range(2, self.kmax + 1)

        for k in ks:
            birch = Birch(n_clusters=k)
            labels = birch.fit_predict(self.x)

            sil_score = silhouette_score(self.x, labels, metric='euclidean')
            db_score = davies_bouldin_score(self.x, labels)
            ch_score = calinski_harabasz_score(self.x, labels)

            self.sil_scores.append(sil_score)
            self.db_scores.append(db_score)
            self.ch_scores.append(ch_score)

            ax = axes[k-2]
            sns.scatterplot(x=self.x.iloc[:, 0], y=self.x.iloc[:, 1], hue=labels, palette='viridis', ax=ax)
            ax.set_title(f'k = {k}\nSilhouette Score: {sil_score:.4f}\nDavies-Bouldin Score: {db_score:.4f}\nCalinski-Harabasz Score: {ch_score:.4f}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

        self.plot_scores(ks)

    def plot_scores(self, ks):
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        sns.lineplot(x=ks, y=self.sil_scores, marker='o', ax=ax[0])
        ax[0].set_title('Silhouette Score')
        ax[0].set_xlabel('Number of clusters (k)')
        ax[0].set_ylabel('Score')

        sns.lineplot(x=ks, y=self.db_scores, marker='o', ax=ax[1])
        ax[1].set_title('Davies-Bouldin Score')
        ax[1].set_xlabel('Number of clusters (k)')
        ax[1].set_ylabel('Score')

        sns.lineplot(x=ks, y=self.ch_scores, marker='o', ax=ax[2])
        ax[2].set_title('Calinski-Harabasz Score')
        ax[2].set_xlabel('Number of clusters (k)')
        ax[2].set_ylabel('Score')

        plt.tight_layout()
        st.pyplot(fig)

        self.display_optimal_k(ks)

    def display_optimal_k(self, ks):
        opt_silhouette_k = ks[np.argmax(self.sil_scores)]
        opt_db_index_k = ks[np.argmin(self.db_scores)]
        opt_chi_index_k = ks[np.argmax(self.ch_scores)]

        st.write(f"Optimal k based on Silhouette Score: {opt_silhouette_k}")
        st.write(f"Optimal k based on Davies-Bouldin Index: {opt_db_index_k}")
        st.write(f"Optimal k based on Calinski-Harabasz Index: {opt_chi_index_k}")

# Usage
# app = BirchClusteringApp(df, kmax=10)
# app.run_clustering()
