"""Integration tests for CrawleoCrawler tool."""

from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_crawleo.crawleo_crawler import CrawleoCrawler


class TestCrawleoCrawlerToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[CrawleoCrawler]:
        return CrawleoCrawler

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the CrawleoCrawler tool
        return {
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

    def test_crawler_with_markdown(self) -> None:
        """Test CrawleoCrawler with markdown output."""
        tool = CrawleoCrawler(markdown=True)
        result = tool.invoke({"urls": ["https://example.com"]})
        assert isinstance(result, dict)

    def test_crawler_multiple_urls(self) -> None:
        """Test CrawleoCrawler with multiple URLs."""
        tool = CrawleoCrawler()
        result = tool.invoke({
            "urls": [
                "https://example.com",
                "https://httpbin.org/html"
            ]
        })
        assert isinstance(result, dict)

