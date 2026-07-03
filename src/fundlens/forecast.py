"""LSTM-based NAV forecaster.

Requires the [forecast] extra: pip install "fundlens[forecast]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class NavForecaster:
    """Trains a small LSTM on a scheme's NAV history and projects future values.

    Parameters
    ----------
    scheme : Scheme
        The fund scheme to forecast.
    lookback : int
        Sliding-window size (number of past trading days) fed to the LSTM.
    """

    def __init__(self, scheme, lookback: int = 60):
        self._scheme = scheme
        self.lookback = lookback
        self._model = None
        self._min: float = 0.0
        self._max: float = 1.0

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _scale(self, arr):
        import numpy as np
        a = np.asarray(arr, dtype=float)
        self._min = float(a.min())
        self._max = float(a.max())
        return (a - self._min) / (self._max - self._min + 1e-8)

    def _unscale(self, arr):
        import numpy as np
        return np.asarray(arr, dtype=float) * (self._max - self._min + 1e-8) + self._min

    def _make_sequences(self, scaled):
        import numpy as np
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback: i])
            y.append(scaled[i])
        return np.array(X)[..., np.newaxis], np.array(y)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, epochs: int = 50, batch_size: int = 32) -> "NavForecaster":
        """Train the LSTM on the scheme's full NAV history.

        Parameters
        ----------
        epochs : int
            Training epochs (50 is usually sufficient for small datasets).
        batch_size : int
            Mini-batch size.

        Returns
        -------
        self
        """
        import tensorflow as tf

        df = self._scheme.nav_history()
        if len(df) <= self.lookback:
            raise ValueError(
                f"Need more than {self.lookback} NAV points to train; "
                f"scheme '{self._scheme.scheme_name}' has only {len(df)}. "
                "Lower lookback or use a scheme with more history."
            )

        scaled = self._scale(df["nav"].values)
        X, y = self._make_sequences(scaled)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback, 1)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self._model = model
        return self

    def predict(self, days: int = 30) -> "pd.Series":
        """Forecast the next *days* NAV values.

        Uses autoregressive prediction: each new forecast is fed back as input
        for the next step, so uncertainty compounds over longer horizons.

        Parameters
        ----------
        days : int
            Number of future trading days to forecast.

        Returns
        -------
        pandas.Series
            NAV forecast indexed by future business dates.
        """
        import numpy as np
        import pandas as pd

        if self._model is None:
            raise RuntimeError("Call .fit() before .predict()")

        df = self._scheme.nav_history()
        scaled = (df["nav"].values - self._min) / (self._max - self._min + 1e-8)
        window = list(scaled[-self.lookback:])

        preds_scaled = []
        for _ in range(days):
            x = np.array(window[-self.lookback:], dtype=float)[np.newaxis, :, np.newaxis]
            step = float(self._model.predict(x, verbose=0)[0, 0])
            preds_scaled.append(step)
            window.append(step)

        preds = self._unscale(preds_scaled)
        last_date = df.index.max()
        future_dates = pd.bdate_range(start=last_date, periods=days + 1)[1:]
        return pd.Series(preds, index=future_dates, name="nav_forecast")
