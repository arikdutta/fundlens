"""Data models returned by :class:`fundlens.FundClient`.

The models are deliberately light. ``Scheme`` carries the raw NAV history and
metadata, plus analytics helpers (``nav_history``, ``returns``, ``volatility``,
``drawdown``) that build pandas/numpy objects lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional


def _parse_date(s: str) -> date:
    """MFApi.in returns dates as DD-MM-YYYY."""
    return datetime.strptime(s, "%d-%m-%Y").date()


@dataclass(frozen=True)
class SearchResult:
    """A single hit from :meth:`FundClient.search`."""

    scheme_code: int
    scheme_name: str

    @classmethod
    def from_api(cls, d: dict) -> "SearchResult":
        return cls(
            scheme_code=int(d["schemeCode"]),
            scheme_name=str(d["schemeName"]),
        )


@dataclass(frozen=True)
class SchemeMeta:
    """Metadata block describing a scheme."""

    scheme_code: int
    scheme_name: str
    fund_house: Optional[str] = None
    scheme_type: Optional[str] = None
    scheme_category: Optional[str] = None
    isin_growth: Optional[str] = None
    isin_div_reinvestment: Optional[str] = None

    @classmethod
    def from_api(cls, d: dict) -> "SchemeMeta":
        return cls(
            scheme_code=int(d.get("scheme_code") or d.get("schemeCode")),
            scheme_name=str(d.get("scheme_name") or d.get("schemeName", "")),
            fund_house=d.get("fund_house"),
            scheme_type=d.get("scheme_type"),
            scheme_category=d.get("scheme_category"),
            isin_growth=d.get("isin_growth"),
            isin_div_reinvestment=d.get("isin_div_reinvestment"),
        )


@dataclass(frozen=True)
class NavPoint:
    """A single NAV observation."""

    nav_date: date
    nav: float

    @classmethod
    def from_api(cls, d: dict) -> "NavPoint":
        return cls(nav_date=_parse_date(d["date"]), nav=float(d["nav"]))


@dataclass
class Scheme:
    """A full scheme: metadata plus its NAV history.

    Analytics helpers require pandas + numpy. They are imported inside the
    methods so that merely importing fundlens (or doing a search) does not
    require the scientific stack.
    """

    meta: SchemeMeta
    nav_points: List[NavPoint] = field(default_factory=list)

    # -- convenience passthroughs -------------------------------------
    @property
    def scheme_code(self) -> int:
        return self.meta.scheme_code

    @property
    def scheme_name(self) -> str:
        return self.meta.scheme_name

    @property
    def latest_nav(self) -> Optional[NavPoint]:
        return self.nav_points[0] if self.nav_points else None

    @classmethod
    def from_api(cls, payload: dict) -> "Scheme":
        meta = SchemeMeta.from_api(payload.get("meta", {}))
        points = [NavPoint.from_api(x) for x in payload.get("data", [])]
        # MFApi returns newest-first; keep that but guard against unsorted data
        points.sort(key=lambda p: p.nav_date, reverse=True)
        return cls(meta=meta, nav_points=points)

    # -- analytics ----------------------------------------------------
    def nav_history(self):
        """Return NAV history as a pandas DataFrame indexed by date (ascending)."""
        import pandas as pd

        if not self.nav_points:
            return pd.DataFrame(columns=["nav"]).rename_axis("date")

        df = pd.DataFrame(
            {"date": [p.nav_date for p in self.nav_points],
             "nav": [p.nav for p in self.nav_points]}
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df

    def returns(self, periods: Optional[dict] = None) -> dict:
        """Trailing point-to-point returns.

        Parameters
        ----------
        periods : dict, optional
            Mapping of label -> number of calendar days. Defaults to common
            trailing windows (1M, 3M, 6M, 1Y, 3Y, 5Y) plus since-inception.

        Returns
        -------
        dict of label -> return as a decimal fraction (0.12 == +12%).
        Windows with insufficient history are omitted.
        """
        import pandas as pd

        df = self.nav_history()
        if df.empty:
            return {}

        if periods is None:
            periods = {"1M": 30, "3M": 91, "6M": 182,
                       "1Y": 365, "3Y": 1095, "5Y": 1825}

        latest_date = df.index.max()
        latest_nav = float(df.loc[latest_date, "nav"])
        out = {}

        for label, days in periods.items():
            target = latest_date - pd.Timedelta(days=days)
            past = df[df.index <= target]
            if past.empty:
                continue
            past_nav = float(past.iloc[-1]["nav"])
            if past_nav <= 0:
                continue
            total = latest_nav / past_nav - 1.0
            # Annualize windows longer than a year (CAGR)
            years = days / 365.25
            if years > 1.0:
                out[label] = (1.0 + total) ** (1.0 / years) - 1.0
            else:
                out[label] = total

        # Since inception (CAGR)
        first_nav = float(df.iloc[0]["nav"])
        if first_nav > 0:
            span_years = (latest_date - df.index.min()).days / 365.25
            if span_years > 0:
                out["since_inception_cagr"] = (
                    (latest_nav / first_nav) ** (1.0 / span_years) - 1.0
                )
        return out

    def volatility(self, annualize: bool = True) -> Optional[float]:
        """Annualized volatility of daily NAV returns (std dev)."""
        import numpy as np

        df = self.nav_history()
        if len(df) < 3:
            return None
        daily = df["nav"].pct_change().dropna()
        if daily.empty:
            return None
        vol = float(daily.std())
        if annualize:
            vol *= np.sqrt(252)
        return vol

    def max_drawdown(self) -> Optional[float]:
        """Maximum peak-to-trough drawdown as a negative fraction."""
        df = self.nav_history()
        if df.empty:
            return None
        nav = df["nav"]
        running_max = nav.cummax()
        drawdown = nav / running_max - 1.0
        return float(drawdown.min())

    def forecast(self, days: int = 30, lookback: int = 60, epochs: int = 50):
        """Forecast future NAV values using an LSTM model.

        Requires the [forecast] extra: ``pip install "fundlens[forecast]"``

        Parameters
        ----------
        days : int
            Number of future trading days to project.
        lookback : int
            Sliding-window size fed to the LSTM (must be < number of NAV points).
        epochs : int
            Training epochs.

        Returns
        -------
        pandas.Series
            Forecasted NAV values indexed by future business dates.
        """
        from .forecast import NavForecaster
        return NavForecaster(self, lookback=lookback).fit(epochs=epochs).predict(days=days)

    def __repr__(self) -> str:
        n = len(self.nav_points)
        return (f"Scheme(code={self.scheme_code}, "
                f"name={self.scheme_name!r}, nav_points={n})")
