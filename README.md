# 🦜️🔗 LangChain Crawleo

[![PyPI version](https://badge.fury.io/py/langchain-crawleo.svg)](https://badge.fury.io/py/langchain-crawleo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package contains the LangChain integration with [Crawleo](https://crawleo.dev/) - a privacy-first web search and crawler API.

## Features

- **CrawleoSearch**: Real-time web search with privacy-first approach
- **CrawleoCrawler**: Extract content from URLs with HTML/markdown support
- Full async support
- LangChain tool calling compatible

---

## Installation

```bash
pip install -U langchain-crawleo
```

### Credentials

Set your Crawleo API key as an environment variable:

```python
import getpass
import os

if not os.environ.get("CRAWLEO_API_KEY"):
    os.environ["CRAWLEO_API_KEY"] = getpass.getpass("Crawleo API key:\n")
```

Or set it directly in your shell:

```bash
export CRAWLEO_API_KEY="your-api-key"
```

---

## Crawleo Search

The `CrawleoSearch` tool allows you to perform web searches using Crawleo's privacy-first Search API.

### Instantiation

The tool accepts various parameters during instantiation:

- `max_pages` (optional, int): Max result pages to crawl. Each page costs 1 credit. Default is 1.
- `setLang` (optional, str): Language code for search interface (e.g., "en", "es", "fr"). Default is "en".
- `cc` (optional, str): Country code for search results (e.g., "US", "GB", "DE").
- `geolocation` (optional, str): Geo location for search. Allowed: random, pl, gb, jp, de, fr, es, us. Default is "random".
- `device` (optional, str): Device simulation: "desktop", "mobile", or "tablet". Default is "desktop".
- `enhanced_html` (optional, bool): Return AI-enhanced, cleaned HTML. Default is True.
- `raw_html` (optional, bool): Return original, unprocessed HTML. Default is False.
- `page_text` (optional, bool): Return extracted plain text. Default is False.
- `markdown` (optional, bool): Return content in Markdown format. Default is True.

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

### Invoke directly with args

The Crawleo search tool accepts the following arguments during invocation:

- `query` (required): A natural language search query
- All other parameters can also be set during invocation

**NOTE**: If you set an argument during instantiation, you can override it during invocation.

```python
# Basic query
result = tool.invoke({"query": "What is Crawleo pricing?"})
print(result)
```

### Parameter Override Example

```python
# Instantiate with default country
search = CrawleoSearch(cc="US")

# Override country during invocation
result = search.invoke({"query": "local news", "cc": "DE"})
# This will use cc="DE" instead of "US"
```

### Response Format

The search API returns a JSON response with the following structure:

```json
{
  "status": "success",
  "data": {
    "query": "What is the future of AI?",
    "pages_fetched": 1,
    "time_used": 1.51,
    "pages": {
      "1": {
        "total_results": "About 523,000 results",
        "search_results": [
          {
            "title": "The Future of AI: Trends and Predictions",
            "link": "https://example.com/ai-future",
            "date": "Mar 12, 2024",
            "snippet": "Artificial Intelligence is rapidly evolving...",
            "domain": "example.com"
          }
        ],
        "page_content": {
          "page_ai_enhanced_html": "...",
          "page_text_markdown": "..."
        }
      }
    },
    "credits_used": 1
  }
}
```

---

## Crawleo Crawler

The `CrawleoCrawler` tool allows you to extract content from specific URLs using Crawleo's Crawler API.

### Instantiation

The tool accepts various parameters during instantiation:

- `raw_html` (optional, bool): Whether to return raw HTML content. Default is False.
- `markdown` (optional, bool): Whether to return content in markdown format. Default is False.

```python
from langchain_crawleo import CrawleoCrawler

tool = CrawleoCrawler(
    markdown=True,
    # raw_html=False,
)
```

### Invoke directly with args

The Crawleo crawler tool accepts the following arguments during invocation:

- `urls` (required): A list of URLs to crawl (1-20 URLs)
- `raw_html` (optional): Override raw_html setting
- `markdown` (optional): Override markdown setting

```python
# Crawl a single URL
result = tool.invoke({"urls": ["https://crawleo.dev"]})
print(result)

# Crawl multiple URLs
result = tool.invoke({
    "urls": [
        "https://example.com",
        "https://another-site.com"
    ]
})
```

---

## Agent Tool Calling

You can use Crawleo tools directly with a LangChain agent. This allows the agent to dynamically set arguments.

```python
import datetime
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_crawleo import CrawleoSearch, CrawleoCrawler

# Initialize LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)

# Initialize Crawleo Tools
search_tool = CrawleoSearch(max_pages=1)
crawler_tool = CrawleoCrawler(markdown=True)

# Set up Prompt
today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a helpful research assistant. You can search the web 
    and crawl specific URLs for information. The date today is {today}."""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create an agent that can use tools
tools = [search_tool, crawler_tool]
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

# Create an Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
user_input = "Search for Crawleo features and then crawl their homepage"
response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
print(response)
```

---

## Async Support

Both tools support async operations:

```python
import asyncio
from langchain_crawleo import CrawleoSearch, CrawleoCrawler

async def main():
    # Async search
    search = CrawleoSearch(max_pages=1)
    search_result = await search.ainvoke({"query": "Crawleo features"})
    print(search_result)
    
    # Async crawl
    crawler = CrawleoCrawler(markdown=True)
    crawl_result = await crawler.ainvoke({"urls": ["https://crawleo.dev"]})
    print(crawl_result)

asyncio.run(main())
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CRAWLEO_API_KEY` | Your Crawleo API key (required) | - |
| `CRAWLEO_BASE_URL` | Custom API base URL (optional) | `https://api.crawleo.dev/api/v1` |

---

## Error Handling

The tools provide helpful error messages and suggestions when things go wrong:

```python
from langchain_crawleo import CrawleoSearch

search = CrawleoSearch()

try:
    result = search.invoke({"query": "very obscure query with no results"})
except Exception as e:
    print(f"Error: {e}")
    # Error will include suggestions like:
    # "Try increasing max_pages for more results"
    # "Try a different or more specific query"
```

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid or missing parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 429 | Too Many Requests - Rate limit or quota exceeded |
| 500 | Internal Server Error - Something went wrong on the server |

---

## License

MIT License - see [LICENSE](LICENSE) for details.

