"""Portfolio-level analytics across multiple schemes.

Build a weighted portfolio of schemes and compute a blended NAV index,
trailing returns, volatility, and pairwise NAV-return correlations.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple, Union

from .models import Scheme

HoldingsInput = Union[
    Mapping[Scheme, float],
    Iterable[Tuple[Scheme, float]],
]


class Portfolio:
    """A weighted basket of :class:`~fundlens.models.Scheme` objects.

    Parameters
    ----------
    holdings : mapping or iterable of (Scheme, weight)
        Either a ``{Scheme: weight}`` mapping or a list of ``(Scheme, weight)``
        tuples. A list is recommended since ``Scheme`` is mutable. Weights are
        normalized to sum to 1.
    """

    def __init__(self, holdings: HoldingsInput):
        if hasattr(holdings, "items"):
            pairs: List[Tuple[Scheme, float]] = list(holdings.items())
        else:
            pairs = list(holdings)
        if not pairs:
            raise ValueError("holdings must contain at least one scheme")
        total = float(sum(w for _, w in pairs))
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        # store as a list of (scheme, normalized_weight)
        self._holdings: List[Tuple[Scheme, float]] = [
            (s, w / total) for s, w in pairs
        ]

    def _aligned_navs(self):
        """Return a DataFrame of each scheme's NAV, forward-filled and aligned."""
        import pandas as pd

        series = {}
        for scheme, _ in self._holdings:
            df = scheme.nav_history()
            if df.empty:
                continue
            series[scheme.scheme_name] = df["nav"]
        if not series:
            return pd.DataFrame()
        combined = pd.concat(series, axis=1)
        combined = combined.sort_index().ffill().dropna()
        return combined

    def index_series(self, base: float = 100.0):
        """Weighted portfolio value indexed to ``base`` at the first common date."""
        navs = self._aligned_navs()
        if navs.empty:
            return navs
        normalized = navs / navs.iloc[0]
        weights = [self.holdings_by_name()[c] for c in navs.columns]
        port = (normalized * weights).sum(axis=1) * base
        port.name = "portfolio"
        return port

    def holdings_by_name(self) -> Dict[str, float]:
        return {s.scheme_name: w for s, w in self._holdings}

    def returns(self, periods: Dict[str, int] | None = None) -> Dict[str, float]:
        """Trailing returns of the blended portfolio index."""
        import pandas as pd

        idx = self.index_series()
        if idx.empty:
            return {}
        if periods is None:
            periods = {"1M": 30, "3M": 91, "6M": 182,
                       "1Y": 365, "3Y": 1095, "5Y": 1825}
        latest_date = idx.index.max()
        latest = float(idx.loc[latest_date])
        out = {}
        for label, days in periods.items():
            target = latest_date - pd.Timedelta(days=days)
            past = idx[idx.index <= target]
            if past.empty:
                continue
            past_val = float(past.iloc[-1])
            total = latest / past_val - 1.0
            years = days / 365.25
            out[label] = ((1 + total) ** (1 / years) - 1) if years > 1 else total
        return out

    def volatility(self) -> float | None:
        """Annualized volatility of the blended portfolio."""
        import numpy as np

        idx = self.index_series()
        if len(idx) < 3:
            return None
        daily = idx.pct_change().dropna()
        return float(daily.std() * np.sqrt(252)) if not daily.empty else None

    def correlation(self):
        """Pairwise correlation matrix of daily NAV returns across holdings."""
        navs = self._aligned_navs()
        if navs.empty or navs.shape[1] < 2:
            import pandas as pd
            return pd.DataFrame()
        return navs.pct_change().dropna().corr()
