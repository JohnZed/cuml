import cupy as cp
import numpy as np
import cudf
import cuml

from dask.distributed import Client, wait
import dask_ml.model_selection as dcv
from dask_cuda import LocalCUDACluster
from sklearn import datasets

from sklearn.model_selection import train_test_split as sk_train_test_split, \
    GridSearchCV as sk_GridSearchCV

from cuml.experimental.hyperparams.dummy import DummyModel
from cuml.experimental.hyperparams.utils import *
import cuml
import sys

ESTIMATORS = {
    'ridge': (cuml.Ridge(fit_intercept=True, solver="eig"), ["alpha"]),
    'lasso': (cuml.Lasso(fit_intercept=False), ["alpha"]),
    'elastic': (cuml.ElasticNet(fit_intercept=False), ["alpha"]),
    'logistic': (cuml.LogisticRegression(fit_intercept=True, solver='qn'), ["C"]),
    'qn': (cuml.QN(fit_intercept=True, loss="softmax"), ["l1_strength"]),
    'dummy': (DummyModel(order='F'), ["alpha"])
}

#
# Parameters
#
N_FOLDS = 5
SCALE_FACTOR = int(1e2)

def set_rmm():
    import cudf
    cudf.set_allocator("default", pool=True,
                       initial_pool_size=int(1.4e10)) # use allocator


if __name__ == '__main__':
    estimator_name = sys.argv[1]
    print("Running: ", estimator_name)

    # Start one worker per GPU on the local system
    cluster = LocalCUDACluster()
    client = Client(cluster)
    client

    client.run(set_rmm)

    diabetes = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = sk_train_test_split(diabetes.data,
                                                           diabetes.target,
                                                           test_size=0.2)

    dup_data = np.array(np.vstack([X_train]*SCALE_FACTOR))
    dup_train = np.array(np.hstack([y_train]*SCALE_FACTOR))
    dup_test_data = np.array(np.vstack([X_test]*SCALE_FACTOR))
    dup_test_y = np.array(np.hstack([y_test]*SCALE_FACTOR))

    print(f"""Duplicated data in memory: {dup_data.nbytes / 1e6} MB
              (Scale factor {SCALE_FACTOR})""")

    record_train_data = (('fea%d'%i, dup_data[:,i])
                         for i in range(dup_data.shape[1]))
    record_test_data = (('fea%d'%i, dup_test_data[:,i])
                        for i in range(dup_data.shape[1]))

    gdf_data = cudf.DataFrame(record_train_data)
    gdf_train = cudf.DataFrame(dict(train=dup_train))
    gdf_test_data = cudf.DataFrame(record_test_data)
    gdf_test_y = cudf.DataFrame(dict(test=dup_test_y))

    cu_clf, param_names = ESTIMATORS[estimator_name]
    params = {param_names[0]: np.logspace(-3, -1, 10)}

    with timed("Dask-nocache-CV5"):
        cu_grid = dcv.GridSearchCV(cu_clf, params, cv=N_FOLDS, cache_cv=False)
        res = cu_grid.fit(gdf_data, gdf_train.train)
        score = cu_grid.score(gdf_test_data, gdf_test_y.test)
        print("Best estimator was: ", cu_grid.best_estimator_)
        print("Score was: ", score)

    with timed("Dask-cache-CV5"):
        cu_grid = dcv.GridSearchCV(cu_clf, params, cv=N_FOLDS, cache_cv=True)
        res = cu_grid.fit(gdf_data, gdf_train.train)
        print("Best estimator was: ", cu_grid.best_estimator_)


# Tested (pass): ridge
# Tested (fail): lasso, elasticnet
