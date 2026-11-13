"""Leak probe: does the model already know what happened?

No tools, no data in the prompt. We name a stock and a date and ask:
  1. monthly: the closing price on the month's last trading day, and whether the stock rose or fell that month;
  2. weekly, inside the paper's window (Q1 2024): did it close higher this Friday than last Friday?
Then we score the answers against the real prices, split at the model's training cutoff.
A model that never saw those dates should score about a coin flip on direction (or the base rate
of up-months). A model that read about them scores much higher: that's the leak.

  python leak_probe.py            # runs every question not yet answered (resumes), then scores
  python leak_probe.py --score    # score only
"""
import argparse
import json
import math
import os
import re

import pandas as pd

import config as C
from data import prices, raw_close

OUT = os.path.join(C.RESULTS, "leak_probe_answers.jsonl")
# Splits inside the probe range: a model may remember the pre-split price.
SPLITS = {"NVDA": [("2024-06-10", 10.0)], "AVGO": [("2024-07-15", 10.0)]}

SYSTEM = ("You are being tested on what you remember from your training data. You have no tools and no internet. "
          "Answer from memory only. If you are unsure, you must still give your best guess: a number and a direction, "
          "never null or unknown. Reply with JSON only.")


def month_ends(df, start, end):
    s = df.loc[f"{start}-01":]
    s = s.loc[: pd.Period(end).end_time]
    return list(s.groupby(s.index.to_period("M")).tail(1).index)


def questions():
    qs = []
    for t in C.PROBE_TICKERS:
        df = prices(t).assign(Close=raw_close(t))   # the close as printed that day
        ends = month_ends(df, C.PROBE_START, C.PROBE_END)
        for d in ends:
            prev = df.loc[: d.replace(day=1) - pd.Timedelta(days=1)].index[-1]   # last trading day of the month before
            qs.append({"kind": "monthly", "ticker": t, "date": d.strftime("%Y-%m-%d"),
                       "prev": prev.strftime("%Y-%m-%d"), "close": float(df.Close[d]), "prev_close": float(df.Close[prev]),
                       "prompt": (f"Stock: {C.NAMES[t]} ({t}), US-listed.\n"
                                  f"1) What was its closing price on {d:%A %d %B %Y}?\n"
                                  f"2) Over the month of {d:%B %Y} (from the close on {prev:%d %B %Y} to the close on {d:%d %B %Y}), "
                                  f"did the stock go up or down?\n"
                                  'Reply as JSON: {"close": <number in USD>, "direction": "up" or "down", "confidence": <0-100>}')})
    for t in C.TICKERS:
        df = prices(t).assign(Close=raw_close(t))
        lo, hi = C.WINDOWS["paper"]
        fridays = df.loc[lo:hi]
        fridays = fridays.groupby(fridays.index.to_period("W-FRI")).tail(1).index
        for d in fridays[1:]:
            prev = df.loc[: d - pd.Timedelta(days=5)].index[-1]
            qs.append({"kind": "weekly_paper", "ticker": t, "date": d.strftime("%Y-%m-%d"),
                       "prev": prev.strftime("%Y-%m-%d"), "close": float(df.Close[d]), "prev_close": float(df.Close[prev]),
                       "prompt": (f"Stock: {C.NAMES[t]} ({t}), US-listed.\n"
                                  f"Did it close higher or lower on {d:%A %d %B %Y} than on {prev:%A %d %B %Y}?\n"
                                  f"Also give its closing price on {d:%d %B %Y}.\n"
                                  'Reply as JSON: {"close": <number in USD>, "direction": "up" or "down", "confidence": <0-100>}')})
    return qs


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    d = str(j.get("direction", "")).lower()
    d = "up" if d.startswith("up") or "high" in d else "down" if d.startswith("down") or "low" in d else None
    try:
        close = float(str(j.get("close")).replace("$", "").replace(",", ""))
    except Exception:
        close = None
    return d, close


def run():
    from spend import chat
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["kind"], r["ticker"], r["date"], r["model"]))
    qs = [q for q in questions() if (q["kind"], q["ticker"], q["date"], C.MODEL) not in done]
    print(f"leak probe: {len(qs)} questions to ask {C.MODEL}")
    for q in qs:
        text = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["prompt"]}],
                    what=f"leak_probe {q['kind']} {q['ticker']} {q['date']}", max_tokens=120)
        with open(OUT, "a") as f:
            f.write(json.dumps({**{k: v for k, v in q.items() if k != "prompt"}, "model": C.MODEL, "answer": text}) + "\n")


def price_hit(t, date, guess, actual, tol=0.05):
    if guess is None or guess <= 0:
        return False
    targets = [actual]
    for sd, f in SPLITS.get(t, []):
        if date < sd:
            targets.append(actual * f)
    return any(abs(guess / a - 1) <= tol for a in targets)


def binom_p(k, n, p=0.5):
    """One-sided P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def score(model=None):
    model = model or C.MODEL
    rows = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    rows = [r for r in rows if r["model"] == model]
    cutoff = C.KNOWLEDGE_CUTOFF   # YYYY-MM
    out = {"model": model, "documented_cutoff": cutoff, "groups": {}}
    by = {}
    for r in rows:
        d, close = parse(r["answer"])
        actual_dir = "up" if r["close"] > r["prev_close"] else "down"
        side = "pre_cutoff" if cutoff and r["date"][:7] <= cutoff else "post_cutoff"
        g = f"{r['kind']}:{side}"
        by.setdefault(g, []).append({"dir_hit": d == actual_dir, "answered": d is not None,
                                     "price_hit": price_hit(r["ticker"], r["date"], close, r["close"]),
                                     "up": actual_dir == "up", "quarter": str(pd.Period(r["date"], "Q"))})
    for g, xs in sorted(by.items()):
        n, k = len(xs), sum(x["dir_hit"] for x in xs)
        base = max(sum(x["up"] for x in xs), n - sum(x["up"] for x in xs)) / n
        out["groups"][g] = {"n": n, "direction_hit_rate": round(k / n, 3), "always_majority_rate": round(base, 3),
                            "p_vs_coin": round(binom_p(k, n), 4),
                            "price_within_5pct": round(sum(x["price_hit"] for x in xs) / n, 3),
                            "answered": sum(x["answered"] for x in xs),
                            "direction_hit_rate_answered": round(k / max(1, sum(x["answered"] for x in xs)), 3)}
    q = {}
    for xs in by.values():
        for x in xs:
            q.setdefault(x["quarter"], []).append(x)
    out["by_quarter"] = {k: {"n": len(v), "direction_hit_rate": round(sum(x["dir_hit"] for x in v) / len(v), 3),
                             "price_within_5pct": round(sum(x["price_hit"] for x in v) / len(v), 3)}
                         for k, v in sorted(q.items())}
    safe = model.replace("/", "_").replace(":", "_")
    json.dump(out, open(os.path.join(C.RESULTS, f"leak_probe_{safe}.json"), "w"), indent=1)
    print(json.dumps(out["groups"], indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if not a.score:
        run()
    score()
