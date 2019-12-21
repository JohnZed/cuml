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
from cuml import Ridge as cumlRidge


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
    print(f"""Duplicated data in memory: {dup_data.nbytes / 1e6} MB
              (Scale factor {SCALE_FACTOR})""")

    record_data = (('fea%d'%i, dup_data[:,i]) for i in range(dup_data.shape[1]))
    gdf_data = cudf.DataFrame(record_data)
    gdf_train = cudf.DataFrame(dict(train=dup_train))

    # cu_clf = DummyModel(order='F')
    cu_clf = cumlRidge(fit_intercept=True, solver="eig")
    params = {'alpha': np.logspace(-3, -1, 10)}

    with timed("Dask-nocache-CV5"):
        cu_grid = dcv.GridSearchCV(cu_clf, params, cv=N_FOLDS, cache_cv=False)
        res = cu_grid.fit(gdf_data, gdf_train.train)
        print("Best estimator was: ", cu_grid.best_estimator_)

    with timed("Dask-cache-CV5"):
        cu_grid = dcv.GridSearchCV(cu_clf, params, cv=N_FOLDS, cache_cv=True)
        res = cu_grid.fit(gdf_data, gdf_train.train)
        print("Best estimator was: ", cu_grid.best_estimator_)

