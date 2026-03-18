"""Convert price-change percentages into discrete trading signals."""

from __future__ import annotations


def label_signal(price_change_pct: float, threshold: float = 0.5) -> str:
    """Return ``'buy'``, ``'sell'``, or ``'hold'`` based on *price_change_pct*.

    Parameters
    ----------
    price_change_pct:
        Percentage change of the close price relative to the previous bar.
    threshold:
        Minimum absolute change (in %) to trigger a buy or sell signal.
    """
    if price_change_pct > threshold:
        return "buy"
    if price_change_pct < -threshold:
        return "sell"
    return "hold"


def label_series(
    closes: list[float], threshold: float = 0.5, lookahead: int = 1
) -> list[str]:
    """Label a full series of close prices with forward-looking signals.

    Each label is based on the percentage change from index *i* to
    index *i + lookahead*.  The last *lookahead* rows receive ``'hold'``.
    """
    labels: list[str] = []
    for i in range(len(closes)):
        if i + lookahead >= len(closes) or closes[i] == 0:
            labels.append("hold")
            continue
        pct = (closes[i + lookahead] - closes[i]) / closes[i] * 100
        labels.append(label_signal(pct, threshold))
    return labels
