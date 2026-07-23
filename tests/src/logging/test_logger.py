import logging

from usel.logging.logger import (
    get_console_logger,
    get_file_logger,
    set_debug_mode,
    PerformanceLogger,
)


def test_console_logger():

    logger = get_console_logger("test_logger")

    assert isinstance(logger, logging.Logger)


def test_file_logger(tmp_path):

    file = tmp_path / "test.log"

    logger = get_file_logger("file_logger_test", file)

    logger.info("hello")

    assert file.exists()


def test_debug_mode():

    logger = logging.getLogger("debug_test")

    set_debug_mode(logger, True)

    assert logger.level == logging.DEBUG


def test_performance_logger_checkpoint():

    perf = PerformanceLogger()

    value = perf.checkpoint("start")

    assert value >= 0


def test_performance_logger_report():

    perf = PerformanceLogger()

    perf.checkpoint("test")

    report = perf.report()

    assert "test" in report


def test_performance_logger_reset():

    perf = PerformanceLogger()

    perf.checkpoint("before")

    perf.reset()

    assert perf.report() == {}
