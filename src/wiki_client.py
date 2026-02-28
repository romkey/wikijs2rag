"""
Wiki.js GraphQL API client.

Targets Wiki.js 2.x. Filters to published, non-private pages only.
An API key is optional but required if your wiki restricts guest access
to the GraphQL endpoint.
"""

import logging
import time
from typing import Optional

import httpx

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


class WikiClientError(Exception):
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

        self._client = httpx.Client(headers=headers, timeout=timeout)

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
                    raise WikiClientError(f"GraphQL errors: {data['errors']}")
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
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    logger.warning(
                        "Request error on attempt %d/%d: %s", attempt, self.max_retries, exc
                    )
                    time.sleep(self.retry_delay * attempt)
                    continue
                raise WikiClientError(str(exc)) from exc

        raise WikiClientError("Exceeded max retries")

    def list_public_pages(self) -> list[dict]:
        """Return metadata for all published, non-private pages."""
        data = self._query(_LIST_PAGES_QUERY)
        pages = data["pages"]["list"]
        public = [p for p in pages if p.get("isPublished") and not p.get("isPrivate")]
        logger.info("Found %d public pages out of %d total", len(public), len(pages))
        return public

    def get_page(self, page_id: int) -> Optional[dict]:
        """Fetch full content for a single page by ID."""
        data = self._query(_GET_PAGE_QUERY, {"id": page_id})
        return data["pages"]["single"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
