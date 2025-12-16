"""Unit tests for CrawleoCrawler tool."""

from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_crawleo.crawleo_crawler import CrawleoCrawler


class TestCrawleoCrawlerToolUnit(ToolsUnitTests):
    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        # Patch the validation_environment class method
        patcher = patch(
            "langchain_crawleo._utilities.CrawleoCrawlerAPIWrapper.validate_environment"
        )
        mock_validate = patcher.start()

        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        return mock_validate

    @property
    def tool_constructor(self) -> Type[CrawleoCrawler]:
        return CrawleoCrawler

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the CrawleoCrawler tool
        return {
            "crawleo_api_key": "fake_key_for_testing",
            "markdown": True,
            "raw_html": False,
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"urls": ["https://example.com"]}


class TestCrawleoCrawlerParameterMerging:
    """Test that parameter merging works correctly."""

    @patch("langchain_crawleo._utilities.CrawleoCrawlerAPIWrapper.crawler")
    def test_instantiation_params_are_used(self, mock_crawler: MagicMock) -> None:
        """Test that instantiation parameters are used when not overridden."""
        mock_crawler.return_value = {"results": [{"content": "test"}]}

        # Instantiate with markdown=True
        tool = CrawleoCrawler(
            crawleo_api_key="fake_key",
            markdown=True,
        )

        # Invoke without overriding markdown
        tool.invoke({"urls": ["https://example.com"]})

        # Verify the crawler was called with instantiation values
        mock_crawler.assert_called_once()
        call_kwargs = mock_crawler.call_args[1]
        assert call_kwargs["markdown"] is True

    @patch("langchain_crawleo._utilities.CrawleoCrawlerAPIWrapper.crawler")
    def test_invocation_params_override_instantiation(self, mock_crawler: MagicMock) -> None:
        """Test that invocation parameters override instantiation parameters."""
        mock_crawler.return_value = {"results": [{"content": "test"}]}

        # Instantiate with markdown=True
        tool = CrawleoCrawler(
            crawleo_api_key="fake_key",
            markdown=True,
        )

        # Invoke with markdown=False to override
        tool.invoke({"urls": ["https://example.com"], "markdown": False})

        # Verify the crawler was called with overridden value
        mock_crawler.assert_called_once()
        call_kwargs = mock_crawler.call_args[1]
        assert call_kwargs["markdown"] is False


class TestCrawleoCrawlerURLConversion:
    """Test that URL list conversion works correctly."""

    @patch("langchain_crawleo._utilities.httpx.get")
    def test_urls_converted_to_comma_separated(self, mock_get: MagicMock) -> None:
        """Test that URLs are converted to comma-separated string."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"content": "test"}]}
        mock_get.return_value = mock_response

        tool = CrawleoCrawler(crawleo_api_key="fake_key")
        tool.invoke({"urls": ["https://a.com", "https://b.com"]})

        # Verify the API was called with comma-separated URLs
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert "https://a.com,https://b.com" in str(call_kwargs["params"]["urls"])


class TestCrawleoCrawlerURLValidation:
    """Test URL validation."""

    @patch("langchain_crawleo._utilities.CrawleoCrawlerAPIWrapper.crawler")
    def test_empty_urls_raises_error(self, mock_crawler: MagicMock) -> None:
        """Test that empty URLs list raises an error."""
        mock_crawler.side_effect = ValueError("At least one URL is required")

        tool = CrawleoCrawler(crawleo_api_key="fake_key")
        result = tool.invoke({"urls": []})

        assert "error" in result

    @patch("langchain_crawleo._utilities.CrawleoCrawlerAPIWrapper.crawler")
    def test_too_many_urls_raises_error(self, mock_crawler: MagicMock) -> None:
        """Test that too many URLs raises an error."""
        mock_crawler.side_effect = ValueError("Maximum of 20 URLs allowed per request")

        tool = CrawleoCrawler(crawleo_api_key="fake_key")
        urls = [f"https://example{i}.com" for i in range(25)]
        result = tool.invoke({"urls": urls})

        assert "error" in result

