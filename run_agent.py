"""Run TradingAgents (pinned commit, unmodified) on a grid of (ticker, date) cells.

What we change around it, and why:
- Prices come from our open-data CSVs, not Yahoo: `yfinance.Ticker` is swapped for LocalTicker,
  which also enforces a point-in-time guard (no row after the analysis date ever reaches the model).
- Only the market analyst runs (price + indicators). The news, social and fundamentals vendors serve
  today's feeds, so on a past date they would hand the model the future.
- One cheap model for both of TradingAgents' tiers; every call is priced and capped by spend.SpendGuard.
- Each window/ticker gets its own memory log, so lessons never cross from a later window into an earlier one.

  python run_agent.py --window dev --tickers AAPL --limit 2     # shake-down
  python run_agent.py --window paper
  python run_agent.py --window holdout                          # sealed: runs once
Every invocation is logged as a trial in results/trials.jsonl.
"""
import argparse
import datetime
import json
import os
import sys
import time
import traceback

import pandas as pd
import yfinance

import config as C
from data import prices

DECISIONS = os.path.join(C.RESULTS, "decisions.jsonl")
TRIALS = os.path.join(C.RESULTS, "trials.jsonl")
AS_OF = {"date": None}   # the point-in-time guard, set before every decision


class LocalTicker:
    """Stands in for yfinance.Ticker: history() from data/prices, cut at the analysis date."""

    def __init__(self, symbol, *a, **k):
        self.symbol = symbol.upper()

    def history(self, start=None, end=None, period=None, **k):
        try:
            df = prices(self.symbol).copy()
        except SystemExit as e:                                   # a missing file is a tool error, not a crash
            raise RuntimeError(str(e)) from None
        if AS_OF["date"] is not None:
            df = df.loc[: AS_OF["date"]]                      # never past the analysis date
        if start is not None:
            df = df.loc[pd.Timestamp(start):]
        if end is not None:
            df = df.loc[: pd.Timestamp(end) - pd.Timedelta(days=1)]   # yfinance's end is exclusive
        df.index.name = "Date"
        return df

    @property
    def info(self):
        return {"symbol": self.symbol, "shortName": C.NAMES.get(self.symbol, self.symbol),
                "longName": C.NAMES.get(self.symbol, self.symbol), "quoteType": "EQUITY",
                "currency": "USD", "exchange": "NMS"}

    def __getattr__(self, name):
        raise RuntimeError(f"LocalTicker: '{name}' is not available offline")


yfinance.Ticker = LocalTicker
yfinance.Search = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("search disabled offline"))


def decision_dates(ticker, window):
    lo, hi = C.WINDOWS[window]
    days = prices(ticker).loc[lo:hi].index
    return [d.strftime("%Y-%m-%d") for d in days[:: C.EVERY_N_TRADING_DAYS]]


def _wire_provider():
    """fal's OpenRouter route speaks the OpenAI API with the FAL key as a Bearer token, so TradingAgents'
    own "openrouter" provider is pointed at it. Reasoning effort goes in the request body."""
    import langchain_openai
    if C.PROVIDER == "fal":
        os.environ["OPENROUTER_API_KEY"] = os.environ["FAL_KEY"]   # in-process only, never printed
    if C.REASONING_EFFORT:
        orig = langchain_openai.ChatOpenAI.__init__

        def init(self, *a, **k):
            k.pop("temperature", None)                              # reasoning models take no temperature
            k["extra_body"] = {**(k.get("extra_body") or {}), "reasoning": {"effort": C.REASONING_EFFORT}}
            orig(self, *a, **k)
        langchain_openai.ChatOpenAI.__init__ = init


def ta_config(run_dir, ticker, window):
    from tradingagents.default_config import build_default_config
    from spend import BASE_URLS
    cfg = build_default_config()
    cfg.update({
        "llm_provider": "openrouter" if C.PROVIDER == "fal" else C.PROVIDER,
        "deep_think_llm": C.MODEL, "quick_think_llm": C.MODEL,
        "backend_url": os.environ.get("TA_BASE_URL") or BASE_URLS.get(C.PROVIDER),
        "temperature": None if C.REASONING_EFFORT else C.TEMPERATURE,
        "max_tokens": C.MAX_TOKENS, "max_debate_rounds": C.DEBATE_ROUNDS, "max_risk_discuss_rounds": C.RISK_ROUNDS,
        "checkpoint_enabled": False,
        "results_dir": os.path.join(run_dir, "reports"),
        "data_cache_dir": os.path.join(run_dir, "cache"),
        "memory_log_path": os.path.join(run_dir, "memory", f"{window}_{ticker}.md"),
        "data_vendors": {**cfg["data_vendors"], "core_stock_apis": "yfinance", "technical_indicators": "yfinance"},
    })
    return cfg


def done_cells(run_tag):
    if not os.path.exists(DECISIONS):
        return set()
    return {(r["window"], r["ticker"], r["date"]) for r in map(json.loads, open(DECISIONS)) if r["run_tag"] == run_tag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, choices=list(C.WINDOWS))
    ap.add_argument("--tickers", default=",".join(C.TICKERS))
    ap.add_argument("--limit", type=int, default=0, help="first N dates per ticker (shake-down)")
    ap.add_argument("--purpose", default="", help="free text stored with the trial")
    a = ap.parse_args()
    if not C.MODEL:
        sys.exit("set TA_MODEL (and TA_PROVIDER, TA_PRICE_IN, TA_PRICE_OUT, TA_CUTOFF)")

    from spend import SpendGuard, OverBudget, spent, check
    _wire_provider()
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    tickers = a.tickers.split(",")
    run_tag = f"{C.MODEL}|{C.config_hash()}"
    if a.window == "holdout" and not os.environ.get("HOLDOUT_GO"):
        sys.exit("holdout: set HOLDOUT_GO=1 once the budget for the whole window is confirmed")
    if a.window == "holdout":
        prior = [json.loads(l) for l in open(TRIALS)] if os.path.exists(TRIALS) else []
        if any(t["window"] == "holdout" and t.get("finished") for t in prior):
            sys.exit("the holdout has already been run once; it stays sealed")
    os.makedirs(C.RESULTS, exist_ok=True)
    trial = {"t": datetime.datetime.now().isoformat(timespec="seconds"), "window": a.window, "tickers": tickers,
             "model": C.MODEL, "provider": C.PROVIDER, "config_hash": C.config_hash(), "limit": a.limit,
             "every_n": C.EVERY_N_TRADING_DAYS, "purpose": a.purpose, "spent_before": round(spent(), 4)}
    have = done_cells(run_tag)
    n_new = n_fail = 0
    run_dir = os.path.join(C.RESULTS, "runs", C.config_hash())
    try:
        # Date-major order: if the spend cap stops a run, every stock is cut at the same date.
        graphs, guards, cells = {}, {}, []
        for t in tickers:
            dates = decision_dates(t, a.window)
            dates = dates[: a.limit] if a.limit else dates
            guards[t] = SpendGuard(f"tradingagents {a.window} {t}")
            graphs[t] = TradingAgentsGraph(selected_analysts=C.ANALYSTS, config=ta_config(run_dir, t, a.window),
                                           callbacks=[guards[t]])
            cells += [(d, i, t) for i, d in enumerate(dates)]
        for d, _, t in sorted(cells):
            graph, guard = graphs[t], guards[t]
            if (a.window, t, d) in have:
                continue
            check()                                   # stop before a decision we cannot pay for
            AS_OF["date"] = pd.Timestamp(d)
            t0, c0 = time.time(), guard.calls
            try:
                state, rating = graph.propagate(t, d)
                row = {"window": a.window, "ticker": t, "date": d, "rating": str(rating),
                       "final": (state.get("final_trade_decision") or "")[-600:]}
            except OverBudget:
                raise
            except Exception as e:
                n_fail += 1
                row = {"window": a.window, "ticker": t, "date": d, "rating": "ERROR",
                       "error": f"{type(e).__name__}: {e}"[:400]}
                traceback.print_exc()
            row.update({"run_tag": run_tag, "llm_calls": guard.calls - c0, "secs": round(time.time() - t0, 1)})
            with open(DECISIONS, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_new += 1
            print(f"{a.window} {t} {d}: {row['rating']}  ({row['llm_calls']} calls, {row['secs']}s, ${spent():.3f} spent)",
                  flush=True)
        trial["finished"] = True
    except OverBudget as e:
        trial["finished"] = False
        trial["stopped"] = str(e)
        print(e)
    finally:
        trial.update({"decisions": n_new, "errors": n_fail, "spent_after": round(spent(), 4)})
        with open(TRIALS, "a") as f:
            f.write(json.dumps(trial) + "\n")


if __name__ == "__main__":
    main()
