import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class KMeansCluster():
    """
    Class which allows user to fit a K-Means model to the training data and
    subsequently predict labels for the test data. Call `assign()`, `fit()`, and
    `predict()` to perform the respective tasks. Some parameters of note:
    
    @param y_train_pred: predicted labels on training data
    @param y_test_pred: predicted labels on test data
    @param sizes_train_df: pd.DataFrame containing counts of train data by label
    @param sizes_test_df: pd.DataFrame containing counts of test data by label
    """
    def __init__(self, k: int=2, X_train=None, X_test=None):
        self.k = k
        self.model = KMeans(n_clusters=self.k, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.labels_ = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.sizes_train_df = None
        self.sizes_test_df = None
    
    def assign(self, X_train, X_test=None):
        self.X_train = X_train
        self.X_test = X_test
    
    def fit(self, X_train=None):
        if X_train is not None:
            self.assign(X_train)
        
        if self.X_train is not None:
            self.model.fit(self.X_train)
            self.labels_ = self.model.labels_
            self.sizes_train_df = pd.DataFrame.from_dict({
                'CLUSTER': [i for i in range(self.k)],
                'CLUSTER_SIZE': [np.sum(self.labels_ == i) for i in range(self.k)],
            }).set_index('CLUSTER')
        else:
            print(f"Please assign a training set before fitting!")
    
    def predict(self):
        self.fit()
        self.y_train_pred = self.model.predict(self.X_train)
        if self.X_test is not None:
            self.y_test_pred = self.model.predict(self.X_test)
            self.sizes_test_df = pd.DataFrame.from_dict({
                'CLUSTER': [i for i in range(self.k)],
                'CLUSTER_SIZE': [np.sum(self.y_test_pred == i) for i in range(self.k)],
            }).set_index('CLUSTER')
        else:
            self.y_test_pred = None
