"""Integration tests for CrawleoSearch tool."""

from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_crawleo.crawleo_search import CrawleoSearch


class TestCrawleoSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[CrawleoSearch]:
        return CrawleoSearch

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the CrawleoSearch tool
        return {
            "max_pages": 1,
            "count": 10,
            "device": "desktop",
            "get_page_text_markdown": True,
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"query": "what is Python programming language"}

    def test_search_with_country_filter(self) -> None:
        """Test CrawleoSearch with country filter."""
        tool = CrawleoSearch(cc="US")
        result = tool.invoke({"query": "latest tech news"})
        assert isinstance(result, dict)

    def test_search_with_language_filter(self) -> None:
        """Test CrawleoSearch with language filter."""
        tool = CrawleoSearch(setLang="en")
        result = tool.invoke({"query": "AI developments"})
        assert isinstance(result, dict)

