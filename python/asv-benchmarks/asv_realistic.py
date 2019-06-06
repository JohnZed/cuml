import pytest
import time
import numpy
from sklearn import datasets, cluster, metrics, linear_model

def time_basic():
    time.sleep(1)

class KMeansSuite:
    params = [
        [10000],
        [10,500],
        [3,50]
    ]

    def setup_cache(self, n_rows, n_features, n_blobs):
        self.X, self.y = datasets.make_blobs(n_samples=n_rows,
                                             n_features=n_features,
                                             centers=n_blobs,
                                             center_box=(-5, 5))
    
    def time_kmeans(self, n_rows, n_features, n_blobs):
        model = cluster.KMeans(n_clusters=n_blobs)
        model.fit(self.X)

    def track_kmeans_accuracy(self, n_rows, n_features, n_blobs):
        X, y = datasets.make_blobs(n_samples=n_rows,
                                   n_features=n_features,
                                   centers=n_blobs,
                                   center_box=(-5, 5))

        model = cluster.KMeans(n_clusters=n_blobs)
        model.fit(self.X)

        y_pred = model.predict(self.X)
        score = metrics.homogeneity_score(self.y, y_pred)
        return score


class LinearSuite:
    params = [
        [10000],
        [10,500],
    ]

    def setup_cache(self, n_rows, n_features):
        self.X, self.y = datasets.make_regression(n_samples=n_rows,
                                                  n_features=n_features)
    
    def time_lasso(self, n_rows, n_features):
        model = linear_model.Lasso()
        model.fit(self.X, self.y)

    def track_lasso(self, n_rows, n_features):
        model = linear_model.Lasso()
        model.fit(self.X, self.y)

        # XXX Within training sample!
        y_pred = model.predict(self.X)
        score = metrics.r2_score(self.y, y_pred)
        return score
