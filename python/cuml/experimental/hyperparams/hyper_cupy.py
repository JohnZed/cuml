"""
Simple script to do gridsearch with cupy inputs.
Derived from Nanthini's script.
"""

import numpy as np
from cuml import Ridge as cumlRidge
from cuml import LogisticRegression as cumlLogisticRegression

import cudf
import copy

from sklearn import datasets
from sklearn.model_selection import train_test_split as sk_train_test_split, \
    GridSearchCV as sk_GridSearchCV

from cuml.experimental.hyperparams.cu_search import patched_GridSearchCV
from cuml.experimental.hyperparams.dummy import DummyModel

import sklearn
from contextlib import contextmanager
import time

@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))

# Optional debugging wrapper to print input datatypes (off for now)
# class Wrapper(cumlRidge):
#     def fit(self, X, y, **kwargs):
#         # print("X class: ", X.__class__, " C-order? ", X._c_contiguous)
#         return super().fit(X, y, **kwargs)

SCALE_FACTOR = int(1e5)
params = {'alpha': np.logspace(-3, -1, 10)}
N_FOLDS = 5

estimators = {
    "dummy-F": (DummyModel(order='F'),
                {'alpha': np.logspace(-3, -1, 10)}),
    "ridge-F": (cumlRidge(fit_intercept=True,
                          solver="eig",
                          normalize=False),
                {'alpha': np.logspace(-3, -1, 10)}),
    "logistic-F": (cumlLogisticRegression(fit_intercept=True,
                                          solver="eig",
                                          normalize=False),
                   {'alpha': np.logspace(-3, -1, 10)}),
    }

# XXX: note: RandomForestClassifier failed with:
#   - AttributeError: 'RandomForestClassifier' object has no attribute 'dtype'
# XXX: note: KNeighborsClassifier failed with:
#  - get_param_names missing (yields "bad param" error")
#  - "inds" used before definition in nearest_neighbors.py - no gpu array support?

#
# Prep dataset
#
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
input_X = gdf_data.values
input_y = gdf_train.train.values


# Stop spew from input_utils
import warnings
warnings.filterwarnings("ignore")

for name, (cu_clf, params) in estimators.items():
    with timed("%14s-patched_search-C" % name):
        cu_grid = patched_GridSearchCV(cu_clf, params, cv=N_FOLDS)
        cu_grid.fit(input_X, input_y, order='C')

    with timed("%14s-patched_search-F" % name):
        cu_grid = patched_GridSearchCV(cu_clf, params, cv=N_FOLDS)
        cu_grid.fit(input_X, input_y, order='F')

    with timed("%14s-sklearn_search" % name):
        sk_grid = sk_GridSearchCV(cu_clf, params, cv=N_FOLDS)
        sk_grid.fit(input_X, input_y)
