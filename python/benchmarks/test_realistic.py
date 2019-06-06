import pytest
import time
import numpy
from sklearn import datasets, cluster, metrics, linear_model

# Invoke with:
#   pytest --junitxml=test.xml test_realistic.py
#
# Demo 3 different estimators, accuracy + perf
# Demo 1 prim

def generate_random_data_np(n_rows, n_cols):
    X_np = np.random.normal(size=(n_rows, n_cols))
    y_np = np.random.normal(size=(n_rows,))

    return X_np, y_np

@pytest.mark.parametrize("n_rows, n_features, n_blobs",
                         [(10000,10,3),  # narrow
                          (5000,500,3),  # wide
                          (1000,10,50), # many-clust
                         ])
def test_kmeans(n_rows, n_features, n_blobs, benchmark_logger, log_mean_value):
    X, y = datasets.make_blobs(n_samples=n_rows,
                               n_features=n_features,
                               centers=n_blobs,
                               center_box=(-5, 5))

    for _ in benchmark_logger.recommended_reps():
        # This will report the min elapsed time
        model = cluster.KMeans(n_clusters=n_blobs)
        with benchmark_logger:
            model.fit(X)

        y_pred = model.predict(X)
        score = metrics.homogeneity_score(y, y_pred)
        log_mean_value(accuracy=score)


@pytest.mark.parametrize("n_rows, n_features",
                         [(250000,10),  # narrow
                          (50000,500),  # wide
                         ])
def test_lasso(n_rows, n_features, benchmark_logger, log_mean_value):
    X, y = datasets.make_regression(n_samples=n_rows,
                                    n_features=n_features)

    for _ in benchmark_logger.recommended_reps():
        # This will report the min elapsed time
        model = linear_model.Lasso()
        with benchmark_logger:
            model.fit(X, y)

        y_pred = model.predict(X)
        score = metrics.r2_score(y, y_pred)
        log_mean_value(accuracy=score)
