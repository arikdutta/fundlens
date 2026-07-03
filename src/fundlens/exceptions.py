"""Exception types for FundLens."""


class FundLensError(Exception):
    """Base class for all FundLens errors."""


class APIError(FundLensError):
    """Raised when the MFApi.in API returns an unexpected status or payload."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class SchemeNotFoundError(FundLensError):
    """Raised when a requested scheme code does not exist."""


class RateLimitError(APIError):
    """Raised when the API responds with HTTP 429 (too many requests)."""
