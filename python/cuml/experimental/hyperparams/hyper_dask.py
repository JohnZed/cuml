"""PoC demo of dask-ml gridsearch with different estimators and options."""
import cupy as cp
import numpy as np
import cudf
import cuml
import copy

from dask.distributed import Client, wait
import dask_ml.model_selection as dcv
from dask_cuda import LocalCUDACluster
from sklearn import datasets

from sklearn.model_selection import train_test_split as sk_train_test_split, \
    GridSearchCV as sk_GridSearchCV
from cuml.experimental.hyperparams.dummy import DummyModel
import sys
import xgboost as xgb




ESTIMATORS = {
    'ridge': (lambda: cuml.Ridge(fit_intercept=True, solver="eig"), ["alpha"]),
    'lasso': (lambda: cuml.Lasso(fit_intercept=False), ["alpha"]),
    'elastic': (lambda: cuml.ElasticNet(fit_intercept=False), ["alpha"]),
    'logistic': (lambda: cuml.LogisticRegression(fit_intercept=True, solver='qn'), ["C"]),
    'qn': (lambda: cuml.QN(fit_intercept=True, loss="softmax"), ["l1_strength"]),
    'rfr': (lambda: cuml.RandomForestRegressor(), ["min_impurity_decrease"]),
    'xgb': (lambda: xgb.XGBRegressor(n_estimators=10, tree_method="gpu_hist"), ["gamma"]),
    'dummy': (DummyModel(order='F'), ["alpha"])
}

#
# Parameters
#
N_FOLDS = 5
SCALE_FACTOR = int(1e2)
DTYPE = np.float32


import time
from contextlib import contextmanager
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))

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

    dup_data = np.array(np.vstack([X_train]*SCALE_FACTOR), dtype=DTYPE)
    dup_train = np.array(np.hstack([y_train]*SCALE_FACTOR), dtype=DTYPE)
    dup_test_data = np.array(np.vstack([X_test]*SCALE_FACTOR), dtype=DTYPE)
    dup_test_y = np.array(np.hstack([y_test]*SCALE_FACTOR), dtype=DTYPE)

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

    cu_clf_func, param_names = ESTIMATORS[estimator_name]
    cu_clf = cu_clf_func()
    params = {param_names[0]: np.logspace(-3, -1, 10)}

    # Start by computing SOL with no work done beyond model creation+fit+score
    nreps_total = len(list(params.values())[0]) * N_FOLDS
    sub_input_nrows = int(gdf_data.shape[0] * (1.0 - (1/float(N_FOLDS))))
    sub_train_X = gdf_data.iloc[:sub_input_nrows, :]
    sub_train_y = gdf_train.train[:sub_input_nrows]
    sub_test_X = gdf_data.iloc[sub_input_nrows:, :]
    sub_test_y = gdf_train.train[sub_input_nrows:]
    nreps_total = len(list(params.values())[0]) * N_FOLDS
    print(f"Running {nreps_total} reps to cover {N_FOLDS} folds")
    with timed("nodask-SOL"):
        for i in range(nreps_total):
            cu_clf_rep = cu_clf_func()
            cu_clf_rep.fit(sub_train_X, sub_train_y)
            cu_clf_rep.score(sub_test_X, sub_test_y)

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

