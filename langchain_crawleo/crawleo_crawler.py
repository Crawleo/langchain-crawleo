"""Crawleo Crawler tool for LangChain."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_crawleo._utilities import CrawleoCrawlerAPIWrapper


class CrawleoCrawlerInput(BaseModel):
    """Input schema for CrawleoCrawler tool."""

    model_config = ConfigDict(extra="allow")

    urls: List[str] = Field(
        description="""List of URLs to crawl.
        
        Must contain between 1 and 20 URLs.
        Example: ["https://example.com", "https://another-site.com"]
        """
    )
    raw_html: Optional[bool] = Field(
        default=None,
        description="""Whether to return raw HTML content from crawled pages.
        
        Set to True if you need the full HTML structure.
        Default is False.
        """,
    )
    markdown: Optional[bool] = Field(
        default=None,
        description="""Whether to return content in markdown format.
        
        Set to True for cleaner, formatted text output.
        Default is False.
        """,
    )


def _generate_suggestions(params: Dict[str, Any]) -> List[str]:
    """Generate helpful suggestions based on the failed crawl parameters."""
    suggestions = []

    urls = params.get("urls", [])

    if not urls:
        suggestions.append("Provide at least one valid URL to crawl")
    elif len(urls) > 20:
        suggestions.append("Reduce the number of URLs to 20 or fewer")
    else:
        suggestions.append("Verify that the URLs are accessible and valid")
        suggestions.append("Try enabling markdown=True for better content extraction")

    return suggestions


class CrawleoCrawler(BaseTool):  # type: ignore[override]
    """Tool that crawls URLs using the Crawleo Crawler API and returns JSON results.

    Setup:
        Install `langchain-crawleo` and set environment variable `CRAWLEO_API_KEY`.

        ```bash
        pip install -U langchain-crawleo
        export CRAWLEO_API_KEY="your-api-key"
        ```

    Instantiate:

        ```python
        from langchain_crawleo import CrawleoCrawler

        tool = CrawleoCrawler(
            markdown=True,
            # raw_html=False,
        )
        ```

    Invoke directly with args:

        ```python
        tool.invoke({"urls": ["https://crawleo.dev"]})
        ```

        Returns JSON with crawled content from the specified URLs.
    """

    name: str = "crawleo_crawler"
    description: str = (
        "A web crawler that extracts content from specified URLs via Crawleo. "
        "Useful for when you need to extract content from specific web pages. "
        "Supports raw HTML extraction and markdown formatting. "
        "Input should be a list of URLs (1-20 URLs allowed)."
    )

    args_schema: Type[BaseModel] = CrawleoCrawlerInput
    handle_tool_error: bool = True

    raw_html: Optional[bool] = None
    """Whether to return raw HTML content.
    
    Default is False.
    """

    markdown: Optional[bool] = None
    """Whether to return content in markdown format.
    
    Default is False.
    """

    api_wrapper: CrawleoCrawlerAPIWrapper = Field(default_factory=CrawleoCrawlerAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with crawleo_api_key and api_base_url if provided
        if "crawleo_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "crawleo_api_key" in kwargs:
                wrapper_kwargs["crawleo_api_key"] = kwargs["crawleo_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = CrawleoCrawlerAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        urls: List[str],
        raw_html: Optional[bool] = None,
        markdown: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Crawl URLs using the Crawleo Crawler API.

        Args:
            urls: List of URLs to crawl (1-20 URLs)
            raw_html: Whether to return raw HTML content
            markdown: Whether to return content in markdown format
            run_manager: Callback manager for the tool run

        Returns:
            Dict[str, Any]: Crawl results from Crawleo API
        """
        try:
            # Merge instantiation defaults with invocation parameters
            # Invocation parameters take precedence if provided
            raw_results = self.api_wrapper.crawler(
                urls=urls,
                raw_html=raw_html if raw_html is not None else self.raw_html,
                markdown=markdown if markdown is not None else self.markdown,
                **kwargs,
            )

            # Check if results are empty or indicate failure
            results = raw_results.get("results", raw_results.get("data", []))
            if not results:
                crawl_params = {
                    "urls": urls,
                    "raw_html": raw_html if raw_html is not None else self.raw_html,
                    "markdown": markdown if markdown is not None else self.markdown,
                }
                suggestions = _generate_suggestions(crawl_params)

                error_message = (
                    f"No content extracted from URLs: {urls}. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."
                )
                raise ToolException(error_message)

            return raw_results

        except ToolException:
            raise
        except ValueError as e:
            # Re-raise validation errors (like URL count limits)
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        urls: List[str],
        raw_html: Optional[bool] = None,
        markdown: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Crawl URLs using the Crawleo Crawler API asynchronously.

        Args:
            urls: List of URLs to crawl (1-20 URLs)
            raw_html: Whether to return raw HTML content
            markdown: Whether to return content in markdown format
            run_manager: Callback manager for the tool run

        Returns:
            Dict[str, Any]: Crawl results from Crawleo API
        """
        try:
            # Merge instantiation defaults with invocation parameters
            # Invocation parameters take precedence if provided
            raw_results = await self.api_wrapper.acrawler(
                urls=urls,
                raw_html=raw_html if raw_html is not None else self.raw_html,
                markdown=markdown if markdown is not None else self.markdown,
                **kwargs,
            )

            # Check if results are empty or indicate failure
            results = raw_results.get("results", raw_results.get("data", []))
            if not results:
                crawl_params = {
                    "urls": urls,
                    "raw_html": raw_html if raw_html is not None else self.raw_html,
                    "markdown": markdown if markdown is not None else self.markdown,
                }
                suggestions = _generate_suggestions(crawl_params)

                error_message = (
                    f"No content extracted from URLs: {urls}. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."
                )
                raise ToolException(error_message)

            return raw_results

        except ToolException:
            raise
        except ValueError as e:
            # Re-raise validation errors (like URL count limits)
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

