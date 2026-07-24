from usel.utils.timing import Timer, timeit


def test_timer_context():

    with Timer() as timer:
        assert timer.elapsed >= 0


def test_timer_label():

    timer = Timer("test")

    assert timer.label == "test"


def test_timer_decorator():

    @Timer()
    def multiply(a, b):
        return a * b

    result = multiply(3, 4)

    assert result == 12

    assert multiply.last_elapsed >= 0


def test_timeit_decorator():

    @timeit
    def add(a, b):
        return a + b

    result, elapsed = add(2, 3)

    assert result == 5

    assert elapsed >= 0
