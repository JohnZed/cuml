import pytest
import time

@pytest.mark.parametrize("x", [1, 10])
def test_sleep_speed(x, benchmark_logger):
    with benchmark_logger:
        time.sleep(x * 0.1)

@pytest.mark.parametrize("x", [1, 10])
def test_multiple_runs(x, benchmark_logger):
    # Run multiple times to reduce variance
    # Could be replaced with fancier timing
    for _ in benchmark_logger.recommended_reps():
        # This will report the min elapsed time
        with benchmark_logger:
            time.sleep(x * 0.01)

def test_time_and_value(log_value, benchmark_logger):
    with benchmark_logger:
        time.sleep(0.01)

    log_value(score=10, other_value=99)
