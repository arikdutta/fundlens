# FundLens

A Python client and analytics toolkit for **Indian mutual fund data**, powered by the free and open [MFApi.in](https://www.mfapi.in) API.

**No API key. No signup. No auth.** MFApi.in serves NAV history and scheme metadata for 14,000+ Indian mutual fund schemes, updated daily, sourced from AMFI.

## What it does

FundLens has three layers:

- **HTTP client** — fetches scheme search results, full NAV history, and metadata from MFApi.in. Includes in-memory caching, exponential-backoff retries on rate limits and server errors, and typed exceptions.
- **Analytics** — computes trailing returns (CAGR for windows > 1 year), annualised volatility, and max drawdown on a single scheme; blended versions of the same metrics plus a pairwise correlation matrix for a weighted portfolio.
- **LSTM forecasting** — trains a small TensorFlow LSTM on a fund's NAV history and projects future NAV values for a given number of trading days.

## Install

```bash
pip install fundlens              # core client (only needs `requests`)
pip install "fundlens[analytics]" # + pandas/numpy for returns, volatility, portfolio
pip install "fundlens[forecast]"  # + tensorflow for LSTM NAV forecasting
```

From source:

```bash
git clone <your-repo> && cd fundlens
pip install -e ".[dev]"
pytest
```

## Quickstart

```python
from fundlens import FundClient

client = FundClient()

# 1. Search by name
hits = client.search("HDFC Flexi Cap")
for h in hits[:3]:
    print(h.scheme_code, h.scheme_name)

# 2. Pull a scheme's full NAV history + metadata
scheme = client.get_scheme(hits[0].scheme_code)
print(scheme.meta.fund_house, scheme.meta.scheme_category)
print("Latest NAV:", scheme.latest_nav.nav, "on", scheme.latest_nav.nav_date)

# 3. Analytics (needs the [analytics] extra)
df = scheme.nav_history()          # pandas DataFrame, date-indexed, ascending
print(scheme.returns())            # {'1M':.., '1Y':.., '3Y':..(CAGR), ...}
print("Annualized vol:", scheme.volatility())
print("Max drawdown:", scheme.max_drawdown())

# 4. LSTM forecast (needs the [forecast] extra)
forecast = scheme.forecast(days=30)   # pandas Series of projected NAV values
print(forecast)
```

## Portfolio analytics

```python
from fundlens import FundClient
from fundlens.portfolio import Portfolio

client = FundClient()
equity = client.get_scheme(118955)   # e.g. a flexi-cap fund
debt   = client.get_scheme(120503)   # e.g. a corporate bond fund

pf = Portfolio([(equity, 0.7), (debt, 0.3)])   # weights auto-normalize
print(pf.returns())        # blended trailing returns
print(pf.volatility())     # blended annualized volatility
print(pf.index_series())   # weighted value indexed to 100
print(pf.correlation())    # pairwise daily-return correlation matrix
```

## NAV forecasting

```python
from fundlens import FundClient, NavForecaster

client = FundClient()
scheme = client.get_scheme(118955)

# Option A: convenience method on Scheme
forecast = scheme.forecast(days=30, lookback=60, epochs=50)

# Option B: use NavForecaster directly for more control
forecaster = NavForecaster(scheme, lookback=60)
forecaster.fit(epochs=100, batch_size=16)
forecast = forecaster.predict(days=60)

print(forecast)   # date-indexed pandas Series of projected NAV values
```

The model is a 2-layer LSTM (64 units, 0.2 dropout) trained on min-max scaled NAV values using a sliding window of `lookback` trading days. Predictions are autoregressive — each forecasted value is fed back as input for the next step, so uncertainty compounds over longer horizons. Results are indicative, not financial advice.

## API surface

| Method | Description |
|---|---|
| `FundClient.search(query)` | Search schemes by (partial) name |
| `FundClient.list_schemes()` | List all schemes (large; cached) |
| `FundClient.get_scheme(code)` | Full NAV history + metadata |
| `FundClient.get_latest_nav(code)` | Latest NAV + metadata only |
| `Scheme.nav_history()` | NAV as a date-indexed DataFrame |
| `Scheme.returns(periods=None)` | Trailing returns (CAGR for >1y windows) |
| `Scheme.volatility()` | Annualized volatility |
| `Scheme.max_drawdown()` | Max peak-to-trough drawdown |
| `Scheme.forecast(days, lookback, epochs)` | LSTM NAV forecast (requires `[forecast]`) |
| `Portfolio.returns()` | Blended trailing returns |
| `Portfolio.volatility()` | Blended annualized volatility |
| `Portfolio.index_series()` | Weighted portfolio value indexed to 100 |
| `Portfolio.correlation()` | Pairwise daily-return correlation matrix |
| `NavForecaster.fit(epochs, batch_size)` | Train LSTM on scheme NAV history |
| `NavForecaster.predict(days)` | Project future NAV values |

## Design notes

- **Resilient client:** exponential-backoff retries on `429`/`5xx`/connection errors, typed exceptions (`SchemeNotFoundError`, `RateLimitError`, `APIError`), and an in-memory response cache (MFApi updates once daily, so caching is safe and courteous).
- **Lazy scientific imports:** the core client only needs `requests`; `pandas`/`numpy` are imported only when you call analytics methods; `tensorflow` is imported only when you call forecast methods.
- **Tested offline:** the test suite mocks all HTTP with `responses`, so `pytest` runs without network access.

## Running the example

```bash
python example.py
```

Do not run `src/fundlens/__init__.py` directly — it is a package initializer and will fail with an import error when executed as a script.

## Data & attribution

Data comes from [MFApi.in](https://www.mfapi.in), which sources NAVs from AMFI (Association of Mutual Funds in India). This package is unaffiliated with MFApi.in or AMFI. Please cache responses and be courteous to the free service.

## License

MIT
