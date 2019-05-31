"""
Fixtures and utilities for benchmark tests
"""
import pytest
import time
import junit_xml as jxml
import pprint

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

class JUnitCollector:
    """Captures test results and reports out in XML"""
    def __init__(self):
        self.suites = {}

    def collect_result(self, request, key, value):
        """Stores a key=value result for current test specified by 'request'"""
        module = request.node.module.__name__
        name = request.node.name
        if module not in self.suites:
            self.suites[module] = {}
        if name not in self.suites[module]:
            self.suites[module][name] = {}

        self.suites[module][name][key] = value

    def junit_tree(self):
        """Returns junit object representing test results"""
        suite_results = []
        for suite_name, suite_cases_raw in self.suites.items():
            suite_cases_parsed = []
            for k,v in suite_cases_raw.items():
                suite_cases_parsed.append(
                    jxml.TestCase(k, k.split('[')[0], v['time']))

            suite_results.append(
                jxml.TestSuite(suite_name, suite_cases_parsed))
        return suite_results

    def junit_xml(self):
        return jxml.TestSuite.to_xml_string(self.junit_tree())

@pytest.fixture(scope="session")
def _junit_writer(request):
    """Tests should not use this fixture directly"""
    collector = JUnitCollector()
    yield collector
    result_str = pprint.pprint(collector.suites)

    # XXX The real version would write to alog
    print("\n---\n", result_str)
    print("\n---\n", collector.junit_xml())


@pytest.fixture
def benchmark_logger(_junit_writer, request):
    """
    The benchmark_logger fixture provides a context manager that tests can invoke like:
      with benchmark_logger:
        ... do something to measure ..

    The time taken by the context block will be logged as the test's time.
    If multiple results are logged within a single test, the lowest time will be recorded.
    """
    bt = BenchTimer()
    yield bt
    _junit_writer.collect_result(request, "time", bt.get_benchmark_time())


@pytest.fixture
def log_value(_junit_writer, request):
    """
    The log_value fixture provides a callable that tests can invoke like
      def test_foo(log_value):
        log_value(my_key=my_result)
    """
    def _inner_logger(**kwargs):
        for k,v in kwargs.items():
            _junit_writer.collect_result(request, k, v)
    yield _inner_logger
