"""Crawleo Search tool for LangChain."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_crawleo._utilities import CrawleoSearchAPIWrapper


class CrawleoSearchInput(BaseModel):
    """Input schema for CrawleoSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Search query to look up (required)")
    max_pages: Optional[int] = Field(
        default=None,
        description="""Maximum number of pages to search.
        
        Each page costs 1 credit. Min: 1.
        Default is 1.
        """,
    )
    setLang: Optional[str] = Field(
        default=None,
        description="""Language code for search interface.
        
        Use standard language codes like "en" for English, "es" for Spanish, "fr" for French.
        Default is "en".
        """,
    )
    cc: Optional[str] = Field(
        default=None,
        description="""Country code for search results.
        
        Use standard country codes like "US" for United States, "GB" for UK, "DE" for Germany.
        Default is None (no country filter).
        """,
    )
    geolocation: Optional[str] = Field(
        default=None,
        description="""Geo location for search.
        
        Allowed values: random, pl, gb, jp, de, fr, es, us.
        Default is "random".
        """,
    )
    device: Optional[Literal["desktop", "mobile", "tablet"]] = Field(
        default=None,
        description="""Device simulation for search.
        
        Options: "desktop", "mobile", or "tablet".
        Default is "desktop".
        """,
    )
    enhanced_html: Optional[bool] = Field(
        default=None,
        description="""Return AI-enhanced, cleaned HTML optimized for processing.
        
        Default is True.
        """,
    )
    raw_html: Optional[bool] = Field(
        default=None,
        description="""Return original, unprocessed HTML of the page.
        
        Default is False.
        """,
    )
    page_text: Optional[bool] = Field(
        default=None,
        description="""Return extracted plain text without HTML tags.
        
        Default is False.
        """,
    )
    markdown: Optional[bool] = Field(
        default=None,
        description="""Return content in Markdown format for easy parsing.
        
        Default is True.
        """,
    )


def _generate_suggestions(params: Dict[str, Any]) -> List[str]:
    """Generate helpful suggestions based on the failed search parameters."""
    suggestions = []

    cc = params.get("cc")
    setLang = params.get("setLang")
    max_pages = params.get("max_pages")

    if cc:
        suggestions.append("Try removing the country filter (cc)")
    if setLang:
        suggestions.append("Try removing the language filter (setLang)")
    if max_pages and max_pages <= 1:
        suggestions.append("Try increasing max_pages for more results")

    if not suggestions:
        suggestions.append("Try a different or more specific query")

    return suggestions


class CrawleoSearch(BaseTool):  # type: ignore[override]
    """Tool that queries the Crawleo Search API and returns JSON results.

    Setup:
        Install `langchain-crawleo` and set environment variable `CRAWLEO_API_KEY`.

        ```bash
        pip install -U langchain-crawleo
        export CRAWLEO_API_KEY="your-api-key"
        ```

    Instantiate:

        ```python
        from langchain_crawleo import CrawleoSearch

        tool = CrawleoSearch(
            max_pages=1,
            cc="US",
            # setLang="en",
            # geolocation="random",
            # device="desktop",
            # enhanced_html=True,
            # raw_html=False,
            # page_text=False,
            # markdown=True,
        )
        ```

    Invoke directly with args:

        ```python
        tool.invoke({"query": "Crawleo pricing"})
        ```

        Returns JSON with search results.
    """

    name: str = "crawleo_search"
    description: str = (
        "A privacy-first web search engine via Crawleo. "
        "Useful for when you need to answer questions about current events or "
        "retrieve real-time information from the web. "
        "Supports multiple languages, country filtering, and various content formats. "
        "Input should be a search query."
    )

    args_schema: Type[BaseModel] = CrawleoSearchInput
    handle_tool_error: bool = True

    max_pages: Optional[int] = None
    """Maximum number of pages to search.
    
    Default is 1.
    """

    setLang: Optional[str] = None
    """Language code for search interface (e.g., "en", "es", "fr").
    
    Default is "en".
    """

    cc: Optional[str] = None
    """Country code for search results (e.g., "US", "GB", "DE").
    
    Default is None (no country filter).
    """

    geolocation: Optional[str] = None
    """Geo location for search.
    
    Allowed: random, pl, gb, jp, de, fr, es, us.
    Default is "random".
    """

    device: Optional[Literal["desktop", "mobile", "tablet"]] = None
    """Device simulation for search.
    
    Default is "desktop".
    """

    enhanced_html: Optional[bool] = None
    """Return AI-enhanced, cleaned HTML optimized for processing.
    
    Default is True.
    """

    raw_html: Optional[bool] = None
    """Return original, unprocessed HTML of the page.
    
    Default is False.
    """

    page_text: Optional[bool] = None
    """Return extracted plain text without HTML tags.
    
    Default is False.
    """

    markdown: Optional[bool] = None
    """Return content in Markdown format for easy parsing.
    
    Default is True.
    """

    api_wrapper: CrawleoSearchAPIWrapper = Field(default_factory=CrawleoSearchAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with crawleo_api_key and api_base_url if provided
        if "crawleo_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "crawleo_api_key" in kwargs:
                wrapper_kwargs["crawleo_api_key"] = kwargs["crawleo_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = CrawleoSearchAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        max_pages: Optional[int] = None,
        setLang: Optional[str] = None,
        cc: Optional[str] = None,
        geolocation: Optional[str] = None,
        device: Optional[Literal["desktop", "mobile", "tablet"]] = None,
        enhanced_html: Optional[bool] = None,
        raw_html: Optional[bool] = None,
        page_text: Optional[bool] = None,
        markdown: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a search query using the Crawleo Search API.

        Args:
            query: Search query to look up
            max_pages: Maximum number of pages to search
            setLang: Language code for search interface
            cc: Country code for search results
            geolocation: Geo location for search
            device: Device simulation for search
            enhanced_html: Return AI-enhanced HTML
            raw_html: Return raw HTML
            page_text: Return plain text
            markdown: Return markdown content
            run_manager: Callback manager for the tool run

        Returns:
            Dict[str, Any]: Search results from Crawleo API
        """
        try:
            # Merge instantiation defaults with invocation parameters
            # Invocation parameters take precedence if provided
            raw_results = self.api_wrapper.search(
                query=query,
                max_pages=max_pages if max_pages is not None else self.max_pages,
                setLang=setLang if setLang is not None else self.setLang,
                cc=cc if cc is not None else self.cc,
                geolocation=geolocation if geolocation is not None else self.geolocation,
                device=device if device is not None else self.device,
                enhanced_html=enhanced_html if enhanced_html is not None else self.enhanced_html,
                raw_html=raw_html if raw_html is not None else self.raw_html,
                page_text=page_text if page_text is not None else self.page_text,
                markdown=markdown if markdown is not None else self.markdown,
                **kwargs,
            )

            # Check if results are empty - response has "data" with "pages"
            data = raw_results.get("data", {})
            pages = data.get("pages", {})
            if not pages:
                search_params = {
                    "cc": cc if cc is not None else self.cc,
                    "setLang": setLang if setLang is not None else self.setLang,
                    "max_pages": max_pages if max_pages is not None else self.max_pages,
                }
                suggestions = _generate_suggestions(search_params)

                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."
                )
                raise ToolException(error_message)

            return raw_results

        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        max_pages: Optional[int] = None,
        setLang: Optional[str] = None,
        cc: Optional[str] = None,
        geolocation: Optional[str] = None,
        device: Optional[Literal["desktop", "mobile", "tablet"]] = None,
        enhanced_html: Optional[bool] = None,
        raw_html: Optional[bool] = None,
        page_text: Optional[bool] = None,
        markdown: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a search query using the Crawleo Search API asynchronously.

        Args:
            query: Search query to look up
            max_pages: Maximum number of pages to search
            setLang: Language code for search interface
            cc: Country code for search results
            geolocation: Geo location for search
            device: Device simulation for search
            enhanced_html: Return AI-enhanced HTML
            raw_html: Return raw HTML
            page_text: Return plain text
            markdown: Return markdown content
            run_manager: Callback manager for the tool run

        Returns:
            Dict[str, Any]: Search results from Crawleo API
        """
        try:
            # Merge instantiation defaults with invocation parameters
            # Invocation parameters take precedence if provided
            raw_results = await self.api_wrapper.asearch(
                query=query,
                max_pages=max_pages if max_pages is not None else self.max_pages,
                setLang=setLang if setLang is not None else self.setLang,
                cc=cc if cc is not None else self.cc,
                geolocation=geolocation if geolocation is not None else self.geolocation,
                device=device if device is not None else self.device,
                enhanced_html=enhanced_html if enhanced_html is not None else self.enhanced_html,
                raw_html=raw_html if raw_html is not None else self.raw_html,
                page_text=page_text if page_text is not None else self.page_text,
                markdown=markdown if markdown is not None else self.markdown,
                **kwargs,
            )

            # Check if results are empty - response has "data" with "pages"
            data = raw_results.get("data", {})
            pages = data.get("pages", {})
            if not pages:
                search_params = {
                    "cc": cc if cc is not None else self.cc,
                    "setLang": setLang if setLang is not None else self.setLang,
                    "max_pages": max_pages if max_pages is not None else self.max_pages,
                }
                suggestions = _generate_suggestions(search_params)

                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."
                )
                raise ToolException(error_message)

            return raw_results

        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

