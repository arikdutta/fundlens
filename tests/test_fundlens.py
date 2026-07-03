"""Tests for FundLens using mocked HTTP responses (no network access needed).

Payload shapes mirror MFApi.in's real responses:
  - search: list of {schemeCode, schemeName}
  - scheme: {meta: {...}, data: [{date: "DD-MM-YYYY", nav: "123.45"}, ...], status}
"""

import datetime as dt

import pytest
import responses

from fundlens import (
    FundClient,
    SchemeNotFoundError,
    RateLimitError,
    APIError,
)
from fundlens.portfolio import Portfolio

BASE = "https://api.mfapi.in"


# ---- fixtures -------------------------------------------------------
def _nav_series(start_nav, days, step):
    """Build a newest-first NAV list like MFApi returns."""
    out = []
    base = dt.date(2024, 1, 1)
    nav = start_nav
    for i in range(days):
        d = base + dt.timedelta(days=i)
        out.append({"date": d.strftime("%d-%m-%Y"), "nav": f"{nav:.4f}"})
        nav *= (1 + step)
    # MFApi returns newest first
    return list(reversed(out))


SCHEME_PAYLOAD = {
    "meta": {
        "fund_house": "HDFC Mutual Fund",
        "scheme_type": "Open Ended",
        "scheme_category": "Equity Scheme - Flexi Cap Fund",
        "scheme_code": 118955,
        "scheme_name": "HDFC Flexi Cap Fund - Growth Option - Direct Plan",
        "isin_growth": "INF179K01YV8",
        "isin_div_reinvestment": None,
    },
    "data": _nav_series(100.0, 400, 0.0005),
    "status": "SUCCESS",
}

SCHEME2_PAYLOAD = {
    "meta": {
        "fund_house": "SBI Mutual Fund",
        "scheme_type": "Open Ended",
        "scheme_category": "Debt Scheme",
        "scheme_code": 120503,
        "scheme_name": "SBI Corporate Bond Fund - Direct Growth",
    },
    "data": _nav_series(50.0, 400, 0.0002),
    "status": "SUCCESS",
}


# ---- search ---------------------------------------------------------
@responses.activate
def test_search_returns_results():
    responses.add(
        responses.GET, f"{BASE}/mf/search",
        json=[
            {"schemeCode": 118955, "schemeName": "HDFC Flexi Cap Fund - Growth"},
            {"schemeCode": 101762, "schemeName": "HDFC Flexi Cap Fund - Growth Plan"},
        ],
        status=200,
    )
    client = FundClient()
    results = client.search("HDFC Flexi")
    assert len(results) == 2
    assert results[0].scheme_code == 118955
    assert "HDFC" in results[0].scheme_name


def test_search_empty_query_raises():
    client = FundClient()
    with pytest.raises(ValueError):
        client.search("   ")


# ---- get_scheme -----------------------------------------------------
@responses.activate
def test_get_scheme_parses_meta_and_navs():
    responses.add(responses.GET, f"{BASE}/mf/118955",
                  json=SCHEME_PAYLOAD, status=200)
    client = FundClient()
    scheme = client.get_scheme(118955)
    assert scheme.scheme_code == 118955
    assert scheme.meta.fund_house == "HDFC Mutual Fund"
    assert len(scheme.nav_points) == 400
    # newest-first ordering preserved
    assert scheme.latest_nav.nav_date > scheme.nav_points[-1].nav_date


@responses.activate
def test_get_scheme_not_found():
    responses.add(responses.GET, f"{BASE}/mf/999999",
                  json={"status": "FAIL", "data": []}, status=200)
    client = FundClient()
    with pytest.raises(SchemeNotFoundError):
        client.get_scheme(999999)


@responses.activate
def test_404_raises_not_found():
    responses.add(responses.GET, f"{BASE}/mf/123", status=404)
    client = FundClient()
    with pytest.raises(SchemeNotFoundError):
        client.get_scheme(123)


# ---- retries / errors ----------------------------------------------
@responses.activate
def test_rate_limit_retries_then_raises():
    for _ in range(4):
        responses.add(responses.GET, f"{BASE}/mf/118955", status=429)
    client = FundClient(max_retries=3, backoff=0.0)
    with pytest.raises(RateLimitError):
        client.get_scheme(118955)


@responses.activate
def test_server_error_then_success():
    responses.add(responses.GET, f"{BASE}/mf/118955", status=500)
    responses.add(responses.GET, f"{BASE}/mf/118955",
                  json=SCHEME_PAYLOAD, status=200)
    client = FundClient(max_retries=3, backoff=0.0)
    scheme = client.get_scheme(118955)
    assert scheme.scheme_code == 118955


@responses.activate
def test_invalid_json_raises_apierror():
    responses.add(responses.GET, f"{BASE}/mf/118955",
                  body="not json", status=200)
    client = FundClient()
    with pytest.raises(APIError):
        client.get_scheme(118955)


# ---- caching --------------------------------------------------------
@responses.activate
def test_cache_avoids_second_call():
    responses.add(responses.GET, f"{BASE}/mf/118955",
                  json=SCHEME_PAYLOAD, status=200)
    client = FundClient(cache=True)
    client.get_scheme(118955)
    client.get_scheme(118955)
    assert len(responses.calls) == 1  # second served from cache


# ---- analytics ------------------------------------------------------
@responses.activate
def test_returns_and_volatility():
    responses.add(responses.GET, f"{BASE}/mf/118955",
                  json=SCHEME_PAYLOAD, status=200)
    scheme = FundClient().get_scheme(118955)

    rets = scheme.returns()
    assert "1M" in rets and "1Y" in rets
    # steady +0.05%/day compounding => positive trailing returns
    assert rets["1M"] > 0
    assert rets["since_inception_cagr"] > 0

    vol = scheme.volatility()
    assert vol is not None and vol >= 0

    dd = scheme.max_drawdown()
    # monotonic uptrend => drawdown ~0
    assert dd is not None and dd <= 0


@responses.activate
def test_nav_history_dataframe_sorted_ascending():
    responses.add(responses.GET, f"{BASE}/mf/118955",
                  json=SCHEME_PAYLOAD, status=200)
    df = FundClient().get_scheme(118955).nav_history()
    assert list(df.columns) == ["nav"]
    assert df.index.is_monotonic_increasing
    assert len(df) == 400


# ---- portfolio ------------------------------------------------------
@responses.activate
def test_portfolio_blended_metrics():
    responses.add(responses.GET, f"{BASE}/mf/118955",
                  json=SCHEME_PAYLOAD, status=200)
    responses.add(responses.GET, f"{BASE}/mf/120503",
                  json=SCHEME2_PAYLOAD, status=200)
    client = FundClient()
    s1 = client.get_scheme(118955)
    s2 = client.get_scheme(120503)

    pf = Portfolio([(s1, 0.6), (s2, 0.4)])
    idx = pf.index_series()
    assert not idx.empty
    assert abs(float(idx.iloc[0]) - 100.0) < 1e-6  # indexed to 100

    rets = pf.returns()
    assert "1Y" in rets
    corr = pf.correlation()
    assert corr.shape == (2, 2)
