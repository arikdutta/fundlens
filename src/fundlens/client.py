"""HTTP client for the MFApi.in mutual fund API.

MFApi.in endpoints (no auth required):
    GET /mf                       -> list all schemes
    GET /mf/search?q=<query>      -> search schemes by name
    GET /mf/<code>                -> full NAV history + meta
    GET /mf/<code>/latest         -> latest NAV + meta
"""

from __future__ import annotations

import time
from typing import List, Optional

import requests

from .exceptions import APIError, RateLimitError, SchemeNotFoundError
from .models import Scheme, SearchResult

DEFAULT_BASE_URL = "https://api.mfapi.in"


class FundClient:
    """Client for MFApi.in.

    Parameters
    ----------
    base_url : str
        API base URL. Override to point at a mirror or a mock server.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Number of retries on 429 / 5xx / connection errors.
    backoff : float
        Base seconds for exponential backoff between retries.
    cache : bool
        If True (default), GET responses are cached in-memory for the life
        of the client. MFApi updates once daily, so caching is safe and
        courteous to the free service.
    session : requests.Session, optional
        Inject your own session (useful for testing or custom adapters).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        cache: bool = True,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._use_cache = cache
        self._cache: dict = {}
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", "fundlens/0.1 (+https://pypi.org/project/fundlens)"
        )

    # -- low-level request -------------------------------------------
    def _get(self, path: str) -> object:
        url = f"{self.base_url}{path}"
        if self._use_cache and url in self._cache:
            return self._cache[url]

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:  # connection-level
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise APIError(f"Request to {url} failed: {exc}") from exc

            if resp.status_code == 429:
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise RateLimitError("Rate limited by MFApi.in (HTTP 429)",
                                     status_code=429)

            if 500 <= resp.status_code < 600:
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise APIError(f"Server error {resp.status_code} from {url}",
                               status_code=resp.status_code)

            if resp.status_code == 404:
                raise SchemeNotFoundError(f"Not found: {url}")

            if resp.status_code != 200:
                raise APIError(f"Unexpected status {resp.status_code} from {url}",
                               status_code=resp.status_code)

            try:
                data = resp.json()
            except ValueError as exc:
                raise APIError(f"Invalid JSON from {url}: {exc}") from exc

            if self._use_cache:
                self._cache[url] = data
            return data

        # Should not reach here, but be safe.
        raise APIError(f"Request to {url} failed after retries: {last_exc}")

    # -- public endpoints --------------------------------------------
    def search(self, query: str) -> List[SearchResult]:
        """Search schemes by (partial) name."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        data = self._get(f"/mf/search?q={requests.utils.quote(query)}")
        if not isinstance(data, list):
            raise APIError("Unexpected search payload (expected a list)")
        return [SearchResult.from_api(d) for d in data]

    def list_schemes(self) -> List[SearchResult]:
        """List all schemes (large payload — cached after first call)."""
        data = self._get("/mf")
        if not isinstance(data, list):
            raise APIError("Unexpected /mf payload (expected a list)")
        return [SearchResult.from_api(d) for d in data]

    def get_scheme(self, scheme_code: int) -> Scheme:
        """Fetch full NAV history + metadata for a scheme code."""
        data = self._get(f"/mf/{int(scheme_code)}")
        if not isinstance(data, dict):
            raise APIError("Unexpected scheme payload (expected an object)")
        # MFApi signals a missing scheme with an empty/`status: FAIL` body
        if data.get("status") == "FAIL" or not data.get("meta"):
            raise SchemeNotFoundError(f"Scheme {scheme_code} not found")
        return Scheme.from_api(data)

    def get_latest_nav(self, scheme_code: int) -> Scheme:
        """Fetch only the latest NAV point + metadata for a scheme."""
        data = self._get(f"/mf/{int(scheme_code)}/latest")
        if not isinstance(data, dict) or not data.get("meta"):
            raise SchemeNotFoundError(f"Scheme {scheme_code} not found")
        return Scheme.from_api(data)

    def clear_cache(self) -> None:
        """Empty the in-memory response cache."""
        self._cache.clear()
