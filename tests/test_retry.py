"""Tests for retry with exponential backoff."""

from unittest.mock import MagicMock, patch

from pygentix.core import Agent
from pygentix.testing import MockAgent


class TestIsRetriable:
    def test_connection_error(self):
        assert Agent.is_retriable(ConnectionError("refused"))

    def test_timeout_error(self):
        assert Agent.is_retriable(TimeoutError("timed out"))

    def test_os_error(self):
        assert Agent.is_retriable(OSError("network unreachable"))

    def test_value_error_not_retriable(self):
        assert not Agent.is_retriable(ValueError("bad input"))

    def test_status_code_429(self):
        exc = Exception("rate limited")
        exc.status_code = 429
        assert Agent.is_retriable(exc)

    def test_status_code_500(self):
        exc = Exception("server error")
        exc.status_code = 500
        assert Agent.is_retriable(exc)

    def test_status_code_200_not_retriable(self):
        exc = Exception("ok?")
        exc.status_code = 200
        assert not Agent.is_retriable(exc)

    def test_status_attribute(self):
        exc = Exception("bad gateway")
        exc.status = 502
        assert Agent.is_retriable(exc)


class TestWithRetry:
    def test_succeeds_first_try(self):
        agent = MockAgent(max_retries=3, retry_delay=0.01)
        fn = MagicMock(return_value="ok")
        assert agent.with_retry(fn) == "ok"
        fn.assert_called_once()

    @patch("time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        agent = MockAgent(max_retries=3, retry_delay=0.01)
        fn = MagicMock(side_effect=[ConnectionError("fail"), "ok"])
        assert agent.with_retry(fn) == "ok"
        assert fn.call_count == 2

    @patch("time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        agent = MockAgent(max_retries=2, retry_delay=0.01)
        fn = MagicMock(side_effect=ConnectionError("always fails"))
        try:
            agent.with_retry(fn)
            assert False, "Expected ConnectionError"
        except ConnectionError:
            pass
        assert fn.call_count == 2

    def test_no_retry_on_non_retriable(self):
        agent = MockAgent(max_retries=3, retry_delay=0.01)
        fn = MagicMock(side_effect=ValueError("bad"))
        try:
            agent.with_retry(fn)
            assert False, "Expected ValueError"
        except ValueError:
            pass
        fn.assert_called_once()

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        agent = MockAgent(max_retries=4, retry_delay=1.0)
        fn = MagicMock(side_effect=[
            ConnectionError("1"), ConnectionError("2"), ConnectionError("3"), "ok",
        ])
        agent.with_retry(fn)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]
