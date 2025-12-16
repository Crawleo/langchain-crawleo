"""Util that calls Crawleo Search + Crawler API.

Crawleo provides privacy-first web search and crawler functionality.
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator


CRAWLEO_API_URL: str = "https://api.crawleo.dev/api/v1"


class CrawleoSearchAPIWrapper(BaseModel):
    """Wrapper for Crawleo Search API."""

    crawleo_api_key: SecretStr
    api_base_url: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        crawleo_api_key = get_from_dict_or_env(
            values, "crawleo_api_key", "CRAWLEO_API_KEY"
        )
        values["crawleo_api_key"] = crawleo_api_key

        # Also allow base URL override from environment
        if "api_base_url" not in values or values.get("api_base_url") is None:
            values["api_base_url"] = os.environ.get("CRAWLEO_BASE_URL", CRAWLEO_API_URL)

        return values

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "x-api-key": self.crawleo_api_key.get_secret_value(),
            "Accept": "application/json",
        }

    def search(
        self,
        query: str,
        max_pages: Optional[int] = None,
        setLang: Optional[str] = None,
        cc: Optional[str] = None,
        geolocation: Optional[str] = None,
        device: Optional[str] = None,
        enhanced_html: Optional[bool] = None,
        raw_html: Optional[bool] = None,
        page_text: Optional[bool] = None,
        markdown: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a search query using the Crawleo Search API.

        Args:
            query: Search query to look up (required)
            max_pages: Maximum number of pages to search. Default is 1.
            setLang: Language code for search interface (e.g., "en", "es", "fr"). Default is "en".
            cc: Country code for search results (e.g., "US", "GB", "DE").
            geolocation: Geo location for search. Allowed: random, pl, gb, jp, de, fr, es, us. Default is "random".
            device: Device simulation: desktop | mobile | tablet. Default is "desktop".
            enhanced_html: Return AI-enhanced, cleaned HTML optimized for processing. Default is True.
            raw_html: Return original, unprocessed HTML of the page. Default is False.
            page_text: Return extracted plain text without HTML tags. Default is False.
            markdown: Return content in Markdown format for easy parsing. Default is True.

        Returns:
            Dict containing search results.
        """
        params: Dict[str, Any] = {"query": query}

        # Add optional parameters if provided
        if max_pages is not None:
            params["max_pages"] = max_pages
        if setLang is not None:
            params["setLang"] = setLang
        if cc is not None:
            params["cc"] = cc
        if geolocation is not None:
            params["geolocation"] = geolocation
        if device is not None:
            params["device"] = device
        if enhanced_html is not None:
            params["enhanced_html"] = enhanced_html
        if raw_html is not None:
            params["raw_html"] = raw_html
        if page_text is not None:
            params["page_text"] = page_text
        if markdown is not None:
            params["markdown"] = markdown

        # Add any extra kwargs
        params.update(kwargs)

        base_url = self.api_base_url or CRAWLEO_API_URL
        response = httpx.get(
            f"{base_url}/search",
            params=params,
            headers=self._get_headers(),
            timeout=60.0,
        )

        if response.status_code != 200:
            try:
                error_detail = response.json()
                error_message = error_detail.get("message", error_detail.get("detail", response.text))
            except Exception:
                error_message = response.text
            raise ValueError(f"Error {response.status_code}: {error_message}")

        return response.json()

    async def asearch(
        self,
        query: str,
        max_pages: Optional[int] = None,
        setLang: Optional[str] = None,
        cc: Optional[str] = None,
        geolocation: Optional[str] = None,
        device: Optional[str] = None,
        enhanced_html: Optional[bool] = None,
        raw_html: Optional[bool] = None,
        page_text: Optional[bool] = None,
        markdown: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a search query using the Crawleo Search API asynchronously.

        Args:
            query: Search query to look up (required)
            max_pages: Maximum number of pages to search. Default is 1.
            setLang: Language code for search interface (e.g., "en", "es", "fr"). Default is "en".
            cc: Country code for search results (e.g., "US", "GB", "DE").
            geolocation: Geo location for search. Allowed: random, pl, gb, jp, de, fr, es, us. Default is "random".
            device: Device simulation: desktop | mobile | tablet. Default is "desktop".
            enhanced_html: Return AI-enhanced, cleaned HTML optimized for processing. Default is True.
            raw_html: Return original, unprocessed HTML of the page. Default is False.
            page_text: Return extracted plain text without HTML tags. Default is False.
            markdown: Return content in Markdown format for easy parsing. Default is True.

        Returns:
            Dict containing search results.
        """
        params: Dict[str, Any] = {"query": query}

        # Add optional parameters if provided
        if max_pages is not None:
            params["max_pages"] = max_pages
        if setLang is not None:
            params["setLang"] = setLang
        if cc is not None:
            params["cc"] = cc
        if geolocation is not None:
            params["geolocation"] = geolocation
        if device is not None:
            params["device"] = device
        if enhanced_html is not None:
            params["enhanced_html"] = enhanced_html
        if raw_html is not None:
            params["raw_html"] = raw_html
        if page_text is not None:
            params["page_text"] = page_text
        if markdown is not None:
            params["markdown"] = markdown

        # Add any extra kwargs
        params.update(kwargs)

        base_url = self.api_base_url or CRAWLEO_API_URL

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/search",
                params=params,
                headers=self._get_headers(),
                timeout=60.0,
            )

            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    error_message = error_detail.get("message", error_detail.get("detail", response.text))
                except Exception:
                    error_message = response.text
                raise ValueError(f"Error {response.status_code}: {error_message}")

            return response.json()


class CrawleoCrawlerAPIWrapper(BaseModel):
    """Wrapper for Crawleo Crawler API."""

    crawleo_api_key: SecretStr
    api_base_url: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        crawleo_api_key = get_from_dict_or_env(
            values, "crawleo_api_key", "CRAWLEO_API_KEY"
        )
        values["crawleo_api_key"] = crawleo_api_key

        # Also allow base URL override from environment
        if "api_base_url" not in values or values.get("api_base_url") is None:
            values["api_base_url"] = os.environ.get("CRAWLEO_BASE_URL", CRAWLEO_API_URL)

        return values

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "x-api-key": self.crawleo_api_key.get_secret_value(),
            "Accept": "application/json",
        }

    def _validate_urls(self, urls: List[str]) -> None:
        """Validate that URLs list is within allowed bounds."""
        if not urls:
            raise ValueError("At least one URL is required")
        if len(urls) > 20:
            raise ValueError("Maximum of 20 URLs allowed per request")

    def crawler(
        self,
        urls: List[str],
        raw_html: Optional[bool] = None,
        markdown: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Crawl URLs using the Crawleo Crawler API.

        Args:
            urls: List of URLs to crawl (required, 1-20 URLs)
            raw_html: Whether to return raw HTML content. Default is False.
            markdown: Whether to return content in markdown format. Default is False.

        Returns:
            Dict containing crawl results.

        Raises:
            ValueError: If URLs list is empty or contains more than 20 URLs.
        """
        self._validate_urls(urls)

        # Join URLs with comma for API parameter
        params: Dict[str, Any] = {"urls": ",".join(urls)}

        # Add optional parameters if provided
        if raw_html is not None:
            params["raw_html"] = raw_html
        if markdown is not None:
            params["markdown"] = markdown

        # Add any extra kwargs
        params.update(kwargs)

        base_url = self.api_base_url or CRAWLEO_API_URL
        response = httpx.get(
            f"{base_url}/crawler",
            params=params,
            headers=self._get_headers(),
            timeout=120.0,  # Longer timeout for crawling
        )

        if response.status_code != 200:
            try:
                error_detail = response.json()
                error_message = error_detail.get("message", error_detail.get("detail", response.text))
            except Exception:
                error_message = response.text
            raise ValueError(f"Error {response.status_code}: {error_message}")

        return response.json()

    async def acrawler(
        self,
        urls: List[str],
        raw_html: Optional[bool] = None,
        markdown: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Crawl URLs using the Crawleo Crawler API asynchronously.

        Args:
            urls: List of URLs to crawl (required, 1-20 URLs)
            raw_html: Whether to return raw HTML content. Default is False.
            markdown: Whether to return content in markdown format. Default is False.

        Returns:
            Dict containing crawl results.

        Raises:
            ValueError: If URLs list is empty or contains more than 20 URLs.
        """
        self._validate_urls(urls)

        # Join URLs with comma for API parameter
        params: Dict[str, Any] = {"urls": ",".join(urls)}

        # Add optional parameters if provided
        if raw_html is not None:
            params["raw_html"] = raw_html
        if markdown is not None:
            params["markdown"] = markdown

        # Add any extra kwargs
        params.update(kwargs)

        base_url = self.api_base_url or CRAWLEO_API_URL

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/crawler",
                params=params,
                headers=self._get_headers(),
                timeout=120.0,  # Longer timeout for crawling
            )

            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    error_message = error_detail.get("message", error_detail.get("detail", response.text))
                except Exception:
                    error_message = response.text
                raise ValueError(f"Error {response.status_code}: {error_message}")

            return response.json()

