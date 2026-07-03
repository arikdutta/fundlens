"""FundLens — a tiny Python client + analytics toolkit for Indian mutual fund data.

Data source: MFApi.in (https://www.mfapi.in) — a free, open REST API for Indian
mutual fund NAV history and scheme metadata. No authentication or API key required.

The client only depends on `requests`. Analytics helpers additionally use
`pandas` and `numpy`, imported lazily so the core client stays lightweight.
"""

from .client import FundClient
from .models import Scheme, SchemeMeta, SearchResult, NavPoint

try:
    from .forecast import NavForecaster
except ImportError:
    NavForecaster = None  # tensorflow not installed; install fundlens[forecast]
from .exceptions import (
    FundLensError,
    SchemeNotFoundError,
    APIError,
    RateLimitError,
)

__version__ = "0.1.0"

__all__ = [
    "FundClient",
    "Scheme",
    "SchemeMeta",
    "SearchResult",
    "NavPoint",
    "NavForecaster",
    "FundLensError",
    "SchemeNotFoundError",
    "APIError",
    "RateLimitError",
    "__version__",
]
