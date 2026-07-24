from usel.utils.profiling import memory_usage_mb, profile


def test_profile_returns_result():

    def add(a, b):
        return a + b

    result, report = profile(add, 5, 10)

    assert result == 15


def test_profile_report_exists():

    def square(x):
        return x * x

    _, report = profile(square, 5)

    assert isinstance(report, str)

    assert len(report) > 0


def test_memory_usage():

    memory = memory_usage_mb()

    assert isinstance(memory, float)

    assert memory >= 0
