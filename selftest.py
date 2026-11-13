"""Check the plumbing with no LLM and no downloads: synthetic prices, random ratings, a fake leak probe.

  python selftest.py
Checks: buy-and-hold through the simulator equals the raw open-to-open move; costs only ever lower
the return; the point-in-time guard refuses dates after the analysis date; the leak-probe scorer
finds a planted leak (perfect before the cutoff, coin flip after) and not a false one.
"""
import json
import os
import random
import tempfile

import numpy as np
import pandas as pd

TMP = tempfile.mkdtemp(prefix="ta_selftest_")
os.environ.update({"DATA_DIR": os.path.join(TMP, "data"), "RESULTS_DIR": os.path.join(TMP, "results"),
                   "TA_MODEL": "selftest", "TA_CUTOFF": "2024-08", "TA_PROVIDER": "ollama"})
os.makedirs(os.path.join(TMP, "data", "prices"))
os.makedirs(os.path.join(TMP, "results"))
idx = pd.bdate_range("2022-06-01", "2026-11-10")
for i, t in enumerate(["AAPL", "GOOGL", "AMZN", "MSFT", "NVDA", "META"]):
    rng = np.random.default_rng(i)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(idx))))
    o = c * np.exp(rng.normal(0, 0.004, len(idx)))
    pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"), "Open": o, "High": np.maximum(o, c) * 1.005,
                  "Low": np.minimum(o, c) * 0.995, "Close": c, "Volume": 1e6}).to_csv(
        os.path.join(TMP, "data", "prices", f"{t}.csv"), index=False)
pd.DataFrame({"observation_date": idx.strftime("%Y-%m-%d"), "DTB3": 5.0}).to_csv(
    os.path.join(TMP, "data", "fred_DTB3.csv"), index=False)

import backtest as B  # noqa: E402
import config as C  # noqa: E402
import leak_probe as L  # noqa: E402
import run_agent as R  # noqa: E402
from data import prices  # noqa: E402

# 1. buy-and-hold golden test
d = R.decision_dates("AAPL", "paper")
dec = [(d[0], "Buy")] + [(x, "Hold") for x in d[1:]]
net = B.simulate("AAPL", dec, "zero")[0]
px = prices("AAPL")
i0 = px.index.searchsorted(pd.Timestamp(d[0]), side="right")
i1 = px.index.searchsorted(pd.Timestamp(d[-1]), side="right")
assert abs((1 + net).prod() - px.Open.iloc[i1] / px.Open.iloc[i0]) < 1e-9, "buy-and-hold mismatch"
# 2. costs only lower the return
random.seed(0)
dec = [(x, random.choice(["Buy", "Sell", "Hold", "Overweight", "Underweight"])) for x in d]
assert (1 + B.simulate("AAPL", dec, "retail")[0]).prod() < (1 + B.simulate("AAPL", dec, "zero")[0]).prod()
# 3. point-in-time guard
R.AS_OF["date"] = pd.Timestamp("2024-01-16")
h = R.LocalTicker("AAPL").history(start="2024-01-01", end="2024-03-30")
assert h.index.max() <= pd.Timestamp("2024-01-16"), "guard let a future row through"
R.AS_OF["date"] = None
# 4. leak-probe scorer: planted leak before the cutoff, none after
with open(L.OUT, "w") as f:
    for q in L.questions():
        up = q["close"] > q["prev_close"]
        if q["date"][:7] > C.KNOWLEDGE_CUTOFF:
            up = random.random() < 0.5
        f.write(json.dumps({**{k: v for k, v in q.items() if k != "prompt"}, "model": "selftest",
                            "answer": json.dumps({"close": q["close"], "direction": "up" if up else "down"})}) + "\n")
s = L.score("selftest")["groups"]
assert s["monthly:pre_cutoff"]["direction_hit_rate"] == 1.0
assert s["monthly:post_cutoff"]["direction_hit_rate"] < 0.7
print("selftest passed:", TMP)
