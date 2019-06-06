"""
Fixtures and utilities for benchmark tests
"""
import pytest
import time
import pprint
from _pytest.junitxml import record_property
from collections import defaultdict
import statistics

class BenchTimer:
    def __init__(self):
        self.results = []

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, typ, val, tb):
        end_time = time.time()
        elapsed = end_time - self.start_time
        self.results.append(elapsed)
        self.start_time = None

    def get_benchmark_time(self):
        """Min time is usually most reliable"""
        return min(self.results)

    def recommended_reps(self):
        """Super-simplistic approach to run many copies of short tests and few copies of long ones"""
        t0 = time.time()
        yield
        elapsed = time.time() - t0
        # Use at least 3 reps, but otherwise try
        # to benchmark for about 5 seconds
        IDEAL_TIME = 5.0
        reps = max(3, min(100, int(IDEAL_TIME / elapsed))) - 1
        for i in range(reps):
            yield




@pytest.fixture
def log_value(request, record_property):
    """
    The log_value fixture provides a callable that tests can invoke like
      def test_foo(log_value):
        log_value(my_key=my_result)
    """
    def _inner_logger(**kwargs):
        for k,v in kwargs.items():
            record_property(k, v)
            
    yield _inner_logger


@pytest.fixture(scope="function")
def log_mean_value(request, record_property):
    """
    The log_mean_value fixture provides a callable that tests can invoke like
      def test_foo(log_mean_value):
        log_mean_value(my_key=my_result)

    This will add the mean value for each key to the output tuple
    if the same key is logged multiple times within a test. E.g.

      for i in range(10):
        log_mean_value(my_num=i)

    will add the "(my_num, 4.5)" tuple to the junit xml
    """
    key_value_results = defaultdict(list)
    
    def _inner_logger(**kwargs):
        for k,v in kwargs.items():
            key_value_results[k].append(v)

    yield _inner_logger

    for k,v in key_value_results.items():
        record_property(k, statistics.mean(v))


@pytest.fixture
def benchmark_logger(request, log_value):
    """
    The benchmark_logger fixture provides a context manager that tests can invoke like:
      with benchmark_logger:
        ... do something to measure ..

    The time taken by the context block will be logged as the test's time.
    If multiple results are logged within a single test, the lowest time will be recorded.
    """
    bt = BenchTimer()
    yield bt
    log_value(benchmark_time=bt.get_benchmark_time())
