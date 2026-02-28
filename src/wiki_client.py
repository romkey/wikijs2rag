"""
Wiki.js GraphQL API client with HTML scraping fallback.

Targets Wiki.js 2.x.

The GraphQL pages.single resolver enforces its own permission layer that
can block guest access even when pages are publicly visible in the browser.
When that happens (error code 6013 / PageViewForbidden) we fall back to
fetching the rendered HTML directly and parsing it with BeautifulSoup.

Priority order for page content:
  1. GraphQL API  (returns clean markdown/raw content + full metadata)
  2. HTML scrape  (returns rendered HTML, converted to plain text)
"""

import logging
import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_LIST_PAGES_QUERY = """
query {
  pages {
    list {
      id
      path
      title
      isPublished
      isPrivate
      contentType
      updatedAt
    }
  }
}
"""

_GET_PAGE_QUERY = """
query GetPage($id: Int!) {
  pages {
    single(id: $id) {
      id
      path
      title
      content
      description
      contentType
      tags {
        tag
      }
      createdAt
      updatedAt
    }
  }
}
"""

# CSS selectors tried in order to find the main content element.
# Wiki.js 2.x renders content inside div.contents; the others are fallbacks
# for customised themes or future versions.
_CONTENT_SELECTORS = [
    "div.contents",
    "div#page-contents",
    "div.page-content",
    "main article",
    "main",
]


class WikiClientError(Exception):
    pass


class WikiPageForbiddenError(WikiClientError):
    """Raised when the API returns error 6013 (PageViewForbidden)."""
    pass


class WikiClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        retry_delay: float = 2.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.graphql_url = f"{self.base_url}/graphql"
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query(self, query: str, variables: Optional[dict] = None) -> dict:
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(self.graphql_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    errors = data["errors"]
                    # Surface forbidden errors as a distinct exception type
                    # so callers can decide whether to try a fallback.
                    for err in errors:
                        code = (err.get("extensions") or {}).get("exception", {}).get("code")
                        msg  = (err.get("message") or "").lower()
                        if code == 6013 or "not authorized" in msg:
                            raise WikiPageForbiddenError(
                                f"GraphQL error 6013 (PageViewForbidden): {errors}"
                            )
                    raise WikiClientError(f"GraphQL errors: {errors}")
                return data["data"]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 or exc.response.status_code >= 500:
                    if attempt < self.max_retries:
                        logger.warning(
                            "HTTP %s on attempt %d/%d, retrying in %.1fs…",
                            exc.response.status_code,
                            attempt,
                            self.max_retries,
                            self.retry_delay,
                        )
                        time.sleep(self.retry_delay * attempt)
                        continue
                raise WikiClientError(str(exc)) from exc
            except WikiPageForbiddenError:
                raise  # don't retry auth failures
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    logger.warning(
                        "Request error on attempt %d/%d: %s", attempt, self.max_retries, exc
                    )
                    time.sleep(self.retry_delay * attempt)
                    continue
                raise WikiClientError(str(exc)) from exc

        raise WikiClientError("Exceeded max retries")

    def _scrape_page(self, path: str, meta: dict) -> Optional[dict]:
        """
        Fetch the rendered HTML for *path* and extract text content.

        Returns a page dict in the same shape as get_page() so callers
        don't need a separate code path, with contentType set to "html".
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        logger.debug("  Falling back to HTML scrape: %s", url)
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WikiClientError(f"HTTP scrape failed for {url}: {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove nav, header, footer, sidebar noise before extracting text
        for tag in soup.select("nav, header, footer, aside, script, style, [role=navigation]"):
            tag.decompose()

        content_el = None
        for selector in _CONTENT_SELECTORS:
            content_el = soup.select_one(selector)
            if content_el:
                logger.debug("  Found content via selector '%s'", selector)
                break

        if not content_el:
            # Last resort: grab the full <body>
            content_el = soup.body
            logger.warning("  No content selector matched for %s; using <body>", path)

        if not content_el:
            return None

        # Preserve paragraph structure for the chunker
        html_content = str(content_el)

        return {
            "id":          meta.get("id"),
            "path":        path,
            "title":       meta.get("title") or "",
            "content":     html_content,
            "description": "",
            "contentType": "html",
            "tags":        [],
            "createdAt":   "",
            "updatedAt":   meta.get("updatedAt") or "",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_public_pages(self) -> list[dict]:
        """Return metadata for all published, non-private pages."""
        data = self._query(_LIST_PAGES_QUERY)
        pages = data["pages"]["list"]
        public = [p for p in pages if p.get("isPublished") and not p.get("isPrivate")]
        logger.info("Found %d public pages out of %d total", len(public), len(pages))
        return public

    def get_page(self, page_id: int, meta: Optional[dict] = None) -> Optional[dict]:
        """
        Fetch full content for a single page.

        Tries the GraphQL API first.  If that returns a 6013 (PageViewForbidden)
        error and *meta* (with at least ``path``) is provided, falls back to
        scraping the rendered HTML.
        """
        try:
            data = self._query(_GET_PAGE_QUERY, {"id": page_id})
            return data["pages"]["single"]
        except WikiPageForbiddenError:
            if meta and meta.get("path"):
                logger.debug(
                    "  GraphQL access denied for page %d, trying HTML scrape.", page_id
                )
                return self._scrape_page(meta["path"], meta)
            raise

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
