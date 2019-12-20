import cuml
from sklearn.base import BaseEstimator
import cupy as cp
from cuml.utils.input_utils import input_to_dev_array

class DummyModel(BaseEstimator):
    """Toy model that does nothing but load the data"""
    def __init__(self, order='F', alpha=1.0):
        self.order = order
        self.alpha = alpha
        pass

    def fit(self, X, y, **kwargs):
        y_m, _, _, _, _ = input_to_dev_array(y, order=self.order)
        X_m, _, _, _, _ = input_to_dev_array(X, order=self.order)
        return self

    def predict(self, X):
        return cp.ones(X.shape[0], dtype='float32')

    def score(self, X, y):
        return 1.0

