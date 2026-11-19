"""
Wiki.js GraphQL API client with HTML scraping fallback.

Targets Wiki.js 2.x. One client is one crawl: anonymous (no API key, so it
sees what a guest sees) or member (an API key scoped to the members group).
Wiki.js applies the credential's permissions to page listing, page content
and uploaded files alike, so a crawl can only see what its tier may see.

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
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Wiki.js GraphQL responses (and the payloads built from them) are untyped JSON.
type JSON = dict[str, Any]

_LIST_PAGES_QUERY = """
query {
  pages {
    list {
      id
      path
      locale
      title
      description
      isPublished
      isPrivate
      contentType
      createdAt
      updatedAt
      tags
    }
  }
}
"""

# Only an administrator (or a full-access API key) may list API keys.
_ADMIN_PROBE_QUERY = """
query {
  authentication {
    apiKeys {
      id
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
      locale
      title
      content
      render
      description
      contentType
      isPublished
      isPrivate
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
        api_key: str | None = None,
        timeout: float = 30.0,
        retry_delay: float = 2.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.graphql_url = f"{self.base_url}/graphql"
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        self.authenticated = bool(api_key)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)
        # Guests carry no credentials at all: no header, and no cookies kept.
        self._guest = httpx.Client(timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query(self, query: str, variables: JSON | None = None) -> JSON:
        payload: JSON = {"query": query}
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
                        msg = (err.get("message") or "").lower()
                        if code == 6013 or "not authorized" in msg:
                            raise WikiPageForbiddenError(f"GraphQL error 6013 (PageViewForbidden): {errors}")
                    raise WikiClientError(f"GraphQL errors: {errors}")
                result: JSON = data["data"]
                return result
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    logger.warning(
                        "HTTP %s on attempt %d/%d, retrying in %.1fs…",
                        exc.response.status_code,
                        attempt,
                        self.max_retries,
                        self.retry_delay * attempt,
                    )
                    time.sleep(self.retry_delay * attempt)
                    continue
                raise WikiClientError(str(exc)) from exc
            except WikiPageForbiddenError:
                raise  # don't retry auth failures
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    logger.warning("Request error on attempt %d/%d: %s", attempt, self.max_retries, exc)
                    time.sleep(self.retry_delay * attempt)
                    continue
                raise WikiClientError(str(exc)) from exc

        raise WikiClientError("Exceeded max retries")

    def _scrape_page(self, path: str, meta: JSON) -> JSON | None:
        """
        Fetch the rendered HTML for *path* and extract text content.

        Returns a page dict in the same shape as get_page() so callers
        don't need a separate code path, with contentType set to "html".
        """
        locale = meta.get("locale") or ""
        url = f"{self.base_url}/{locale + '/' if locale else ''}{path.lstrip('/')}"
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

        html_content = str(content_el)

        return {
            "id": meta.get("id"),
            "path": path,
            "locale": meta.get("locale") or "",
            "title": meta.get("title") or "",
            "content": html_content,
            "render": html_content,
            "description": meta.get("description") or "",
            "contentType": "html",
            "tags": [{"tag": t} for t in meta.get("tags") or []],
            "createdAt": meta.get("createdAt") or "",
            "updatedAt": meta.get("updatedAt") or "",
            "scraped": True,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_pages(self) -> list[JSON]:
        """Metadata for every published, non-private page this crawl's
        credentials can read, in every locale."""
        data = self._query(_LIST_PAGES_QUERY)
        pages: list[JSON] = data["pages"]["list"] or []
        visible = [p for p in pages if p.get("isPublished") and not p.get("isPrivate")]
        logger.info(
            "%s crawl lists %d pages (%d unpublished or private skipped)",
            "Member" if self.authenticated else "Anonymous",
            len(visible),
            len(pages) - len(visible),
        )
        return visible

    def has_admin_rights(self) -> bool:
        """True if these credentials can do something only an administrator
        can. The pipeline must never run with such a credential."""
        try:
            data = self._query(_ADMIN_PROBE_QUERY)
        except WikiClientError:
            return False
        return bool((data.get("authentication") or {}).get("apiKeys") is not None)

    def guest_can_view(self, url: str) -> bool:
        """Whether a guest, with no credentials, gets *url* rather than a login page."""
        try:
            resp = self._guest.get(url)
        except httpx.HTTPError as exc:
            raise WikiClientError(f"Guest check of {url} failed: {exc}") from exc
        return resp.status_code == 200 and not resp.url.path.rstrip("/").endswith("/login")

    def fetch_file(self, url: str, max_bytes: int) -> tuple[bytes, str] | None:
        """(content, media type) of a file on the wiki, fetched with this
        crawl's credentials, or None if it can't be read or is too big."""
        try:
            with self._client.stream("GET", url) as resp:
                if resp.status_code != 200 or resp.url.path.rstrip("/").endswith("/login"):
                    logger.debug("  Can't fetch %s (HTTP %d)", url, resp.status_code)
                    return None
                media_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if media_type == "text/html":
                    return None  # a page, or a login form, not a file
                chunks: list[bytes] = []
                size = 0
                for part in resp.iter_bytes():
                    size += len(part)
                    if size > max_bytes:
                        logger.warning("  Skipping %s: larger than %d bytes", url, max_bytes)
                        return None
                    chunks.append(part)
        except httpx.HTTPError as exc:
            logger.warning("  Can't fetch %s: %s", url, exc)
            return None
        return b"".join(chunks), media_type

    def get_page(self, page_id: int, meta: JSON | None = None) -> JSON | None:
        """
        Fetch full content for a single page.

        Tries the GraphQL API first.  If that returns a 6013 (PageViewForbidden)
        error and *meta* (with at least ``path``) is provided, falls back to
        scraping the rendered HTML.
        """
        try:
            data = self._query(_GET_PAGE_QUERY, {"id": page_id})
            page: JSON | None = data["pages"]["single"]
            return page
        except WikiPageForbiddenError:
            if meta and meta.get("path"):
                logger.debug("  GraphQL access denied for page %d, trying HTML scrape.", page_id)
                return self._scrape_page(meta["path"], meta)
            raise

    def close(self) -> None:
        self._client.close()
        self._guest.close()

    def __enter__(self) -> WikiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
