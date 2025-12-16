"""LangChain integration for Crawleo - Privacy-first web search and crawler."""

from importlib import metadata
from typing import List

from langchain_crawleo.crawleo_search import CrawleoSearch
from langchain_crawleo.crawleo_crawler import CrawleoCrawler

try:
    __version__: str = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__: List[str] = [
    "CrawleoSearch",
    "CrawleoCrawler",
    "__version__",
]

