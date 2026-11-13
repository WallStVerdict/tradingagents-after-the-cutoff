"""Open data only. Daily prices for each stock, the 3-month T-bill rate from FRED.

Prices live in data/prices/<TICKER>.csv with the columns Date,Open,High,Low,Close,Volume
(Stooq's CSV format). Two ways to fill them:
  python data.py --tiingo        # Tiingo's free API, needs TIINGO_API_KEY in the environment
  manual Stooq download          # stooq.com/q/d/?s=aapl.us -> "Download data in csv file", save as data/prices/AAPL.csv
The FRED series download without a key. `python data.py --check` verifies data.sha256.

We do not scrape Yahoo. TradingAgents reads Yahoo by default; run_agent.py swaps that for these files.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request

import pandas as pd

from config import DATA, TICKERS, PROBE_TICKERS

PRICES = os.path.join(DATA, "prices")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2022-01-01"
UA = "tradingagents-after-the-cutoff (research; 351215329+WallStVerdict@users.noreply.github.com)"
START = "2022-06-01"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_fred(sid="DTB3"):
    os.makedirs(DATA, exist_ok=True)
    raw = _get(FRED_URL.format(sid=sid))
    path = os.path.join(DATA, f"fred_{sid}.csv")
    open(path, "wb").write(raw)
    return path


def fetch_tiingo(tickers):
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        sys.exit("set TIINGO_API_KEY (free account at tiingo.com), or download the Stooq CSVs by hand")
    os.makedirs(PRICES, exist_ok=True)
    for t in tickers:
        url = (f"https://api.tiingo.com/tiingo/daily/{t.lower()}/prices?startDate={START}"
               f"&format=json&token={key}")
        rows = json.loads(_get(url))
        # split- and dividend-adjusted prices, so returns include dividends
        df = pd.DataFrame([{"Date": r["date"][:10], "Open": r["adjOpen"], "High": r["adjHigh"],
                            "Low": r["adjLow"], "Close": r["adjClose"], "Volume": r["adjVolume"]} for r in rows])
        df.to_csv(os.path.join(PRICES, f"{t}.csv"), index=False)
        print(f"{t}: {len(df)} rows {df.Date.iloc[0]} .. {df.Date.iloc[-1]}")


def fetch_yahoo(tickers):
    """Yahoo Finance daily bars through yfinance (TradingAgents' own default source).
    Used for episode 3 only, with the user's OK (11 Nov 2026). Open/High/Low/Close are adjusted for splits
    and dividends (auto_adjust=True); RawClose is the close as printed that day (split-adjusted only)."""
    import yfinance as yf
    os.makedirs(PRICES, exist_ok=True)
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    for t in tickers:
        adj = yf.Ticker(t).history(start=START, end=end, auto_adjust=True, actions=False)
        raw = yf.Ticker(t).history(start=START, end=end, auto_adjust=False, actions=False)
        df = adj[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["RawClose"] = raw["Close"].reindex(df.index)
        df.index = df.index.tz_localize(None).strftime("%Y-%m-%d")
        df.index.name = "Date"
        df.round(6).to_csv(os.path.join(PRICES, f"{t}.csv"))
        print(f"{t}: {len(df)} rows {df.index[0]} .. {df.index[-1]}")
    open(os.path.join(DATA, "PULLED.txt"), "w").write(
        f"Yahoo Finance daily bars via yfinance, pulled {pd.Timestamp.now():%Y-%m-%d %H:%M}, start {START}, end {end} (exclusive)\n")


def raw_close(ticker):
    path = os.path.join(PRICES, f"{ticker}.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df["RawClose"] if "RawClose" in df else df["Close"]


def prices(ticker):
    path = os.path.join(PRICES, f"{ticker}.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path}: see the top of data.py")
    df = pd.read_csv(path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def risk_free_daily(index):
    """Daily risk-free return from the 3-month T-bill (FRED DTB3, % a year), aligned to `index`."""
    path = os.path.join(DATA, "fred_DTB3.csv")
    if not os.path.exists(path):
        fetch_fred("DTB3")
    df = pd.read_csv(path)
    df.columns = ["Date", "rate"]
    df["Date"] = pd.to_datetime(df["Date"])
    s = pd.to_numeric(df.set_index("Date")["rate"], errors="coerce").ffill()
    return (s.reindex(index, method="ffill") / 100 / 252).fillna(0.0)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_checksums():
    lines = []
    for root, _, files in os.walk(DATA):
        for f in sorted(files):
            if f.endswith(".csv"):
                p = os.path.join(root, f)
                lines.append(f"{sha256(p)}  {os.path.relpath(p, os.path.dirname(DATA))}")
    open(os.path.join(os.path.dirname(DATA), "data.sha256"), "w").write("\n".join(sorted(lines)) + "\n")
    print("\n".join(sorted(lines)))


def check_checksums():
    base = os.path.dirname(DATA)
    bad = 0
    for line in open(os.path.join(base, "data.sha256")):
        if line.startswith("#") or not line.strip():
            continue
        h, p = line.split()
        full = os.path.join(base, p)
        ok = os.path.exists(full) and sha256(full) == h
        bad += not ok
        print(("ok   " if ok else "DIFF ") + p)
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiingo", action="store_true")
    ap.add_argument("--fred", action="store_true")
    ap.add_argument("--yahoo", action="store_true", help="episode 3 only: Yahoo via yfinance")
    ap.add_argument("--checksums", action="store_true", help="write data.sha256")
    ap.add_argument("--check", action="store_true", help="verify data.sha256")
    a = ap.parse_args()
    if a.yahoo:
        fetch_yahoo(sorted(set(TICKERS + PROBE_TICKERS + ["SPY"])))   # SPY: TradingAgents' benchmark for alpha
    if a.fred or a.tiingo or a.yahoo:
        print(fetch_fred("DTB3"))
    if a.tiingo:
        fetch_tiingo(sorted(set(TICKERS + PROBE_TICKERS)))
    if a.checksums:
        write_checksums()
    if a.check:
        sys.exit(0 if check_checksums() else 1)
