"""Test API key handling."""

from langchain_crawleo._utilities import CrawleoSearchAPIWrapper, CrawleoCrawlerAPIWrapper


def test_search_api_wrapper_api_key_not_visible() -> None:
    """Test that the API key is not visible in CrawleoSearchAPIWrapper repr."""
    wrapper = CrawleoSearchAPIWrapper(crawleo_api_key="abcd123")  # type: ignore[arg-type]
    assert "abcd123" not in repr(wrapper)


def test_crawler_api_wrapper_api_key_not_visible() -> None:
    """Test that the API key is not visible in CrawleoCrawlerAPIWrapper repr."""
    wrapper = CrawleoCrawlerAPIWrapper(crawleo_api_key="abcd123")  # type: ignore[arg-type]
    assert "abcd123" not in repr(wrapper)
