import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class KMeansCluster():
    def __init__(self, k: int=2, X_train=None, X_test=None):
        self.k = k
        self.model = KMeans(n_clusters=self.k, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.labels_ = None
        self.y_train_pred = None
        self.y_test_pred = None
    
    def assign(self, X_train, X_test=None):
        self.X_train = X_train
        self.X_test = X_test
    
    def fit(self, X_train=None):
        if X_train is not None:
            self.assign(X_train)
        
        if self.X_train is not None:
            self.model.fit(self.X_train)
            self.labels_ = self.model.labels_
            self.sizes_df = pd.DataFrame.from_dict({
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
        else:
            self.y_test_pred = None
        return self.y_train_pred, self.y_test_pred
