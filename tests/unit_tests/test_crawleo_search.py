"""Unit tests for CrawleoSearch tool."""

from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_crawleo.crawleo_search import CrawleoSearch


class TestCrawleoSearchToolUnit(ToolsUnitTests):
    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        # Patch the validation_environment class method
        patcher = patch(
            "langchain_crawleo._utilities.CrawleoSearchAPIWrapper.validate_environment"
        )
        mock_validate = patcher.start()

        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        return mock_validate

    @property
    def tool_constructor(self) -> Type[CrawleoSearch]:
        return CrawleoSearch

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the CrawleoSearch tool
        return {
            "crawleo_api_key": "fake_key_for_testing",
            "max_pages": 1,
            "cc": "US",
            "device": "desktop",
            "markdown": True,
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"query": "what is Crawleo?"}


class TestCrawleoSearchParameterMerging:
    """Test that parameter merging works correctly."""

    @patch("langchain_crawleo._utilities.CrawleoSearchAPIWrapper.search")
    def test_instantiation_params_are_used(self, mock_search: MagicMock) -> None:
        """Test that instantiation parameters are used when not overridden."""
        mock_search.return_value = {"status": "success", "data": {"pages": {"1": {}}}}

        # Instantiate with cc="US"
        tool = CrawleoSearch(
            crawleo_api_key="fake_key",
            cc="US",
            max_pages=2,
        )

        # Invoke without overriding cc
        tool.invoke({"query": "test query"})

        # Verify the search was called with instantiation values
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["cc"] == "US"
        assert call_kwargs["max_pages"] == 2

    @patch("langchain_crawleo._utilities.CrawleoSearchAPIWrapper.search")
    def test_invocation_params_override_instantiation(self, mock_search: MagicMock) -> None:
        """Test that invocation parameters override instantiation parameters."""
        mock_search.return_value = {"status": "success", "data": {"pages": {"1": {}}}}

        # Instantiate with cc="US"
        tool = CrawleoSearch(
            crawleo_api_key="fake_key",
            cc="US",
        )

        # Invoke with cc="DE" to override
        tool.invoke({"query": "test query", "cc": "DE"})

        # Verify the search was called with overridden value
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["cc"] == "DE"

    @patch("langchain_crawleo._utilities.CrawleoSearchAPIWrapper.search")
    def test_none_invocation_param_uses_instantiation_default(self, mock_search: MagicMock) -> None:
        """Test that None invocation params use instantiation defaults."""
        mock_search.return_value = {"status": "success", "data": {"pages": {"1": {}}}}

        tool = CrawleoSearch(
            crawleo_api_key="fake_key",
            max_pages=3,
        )

        # Invoke without max_pages - should use instantiation default
        tool.invoke({"query": "test query"})

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["max_pages"] == 3

