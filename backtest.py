"""Turn TradingAgents' ratings into a position, a P&L after costs, and compare with two baselines.

Rules (written down before any result was seen):
- Buy / Overweight = long 1x. Sell / Underweight = short 1x (the paper's Fig. 6 shows long and short).
  Hold = keep the current position. The book starts flat. REVIEW / ERROR = keep the current position.
- A decision made on day D (using data up to D's close) is filled at the next trading day's open.
- The position is held open-to-open until the next fill. The window ends at the open after the last day.
- Costs: COSTS[preset]["bps"] per unit of position traded (a long-to-short flip trades 2 units),
  plus a borrow fee on short days.
- Buy-and-hold: long from the first fill to the end, one entry and one exit charged.
- Random agent: same decision dates, same number of position changes, random direction at each change,
  RANDOM_SEEDS seeds. We report the median and where TradingAgents ranks among them.
Metrics follow the paper's appendix S1.2: CR, AR, Sharpe (annualised, excess over 3-month T-bills), MDD.
"""
import json
import os

import numpy as np
import pandas as pd

import config as C
from data import prices, risk_free_daily

DECISIONS = os.path.join(C.RESULTS, "decisions.jsonl")
LONG = {"buy", "overweight"}
SHORT = {"sell", "underweight"}


def target(rating, current):
    """A rating to a position. Ints pass straight through (the random agent uses -1, 0, 1)."""
    if isinstance(rating, (int, np.integer)):
        return int(rating)
    r = str(rating).strip().lower()
    return 1 if r in LONG else -1 if r in SHORT else current


def open_returns(ticker, first_date, last_date):
    """Open-to-open daily returns from the first fill (day after first_date) to the open after last_date."""
    px = prices(ticker)
    idx = px.index
    i0 = idx.searchsorted(pd.Timestamp(first_date), side="right")
    i1 = idx.searchsorted(pd.Timestamp(last_date), side="right")
    opens = px.Open.iloc[i0: i1 + 1]
    return opens.pct_change().dropna(), opens.index


def simulate(ticker, decisions, cost="retail"):
    """decisions: list of (date, rating) sorted. Returns daily net returns and the trade count."""
    c = C.COSTS[cost]
    dates = [pd.Timestamp(d) for d, _ in decisions]
    rets, open_days = open_returns(ticker, dates[0], dates[-1])
    px_idx = prices(ticker).index
    # position held during the day that starts at each open (index = start-of-period open date)
    pos_by_fill = {}
    cur = 0
    for d, rating in decisions:
        new = target(rating, cur)
        fill = px_idx[px_idx.searchsorted(pd.Timestamp(d), side="right")]
        pos_by_fill[fill] = new
        cur = new
    starts = open_days[:-1]
    pos = pd.Series(np.nan, index=starts)
    for f, p in pos_by_fill.items():
        if f in pos.index:
            pos[f] = p
    pos = pos.ffill().fillna(0.0)
    gross = pos.values * rets.values
    traded = np.abs(np.diff(np.concatenate([[0.0], pos.values, [0.0]])))   # entry ... final exit
    cost_per_day = traded[:-1] * c["bps"] / 1e4
    cost_per_day[-1] += traded[-1] * c["bps"] / 1e4
    borrow = np.where(pos.values < 0, c["borrow_pa"] / 252, 0.0)
    net = pd.Series(gross - cost_per_day - borrow, index=rets.index)
    n_changes = int((np.diff(np.concatenate([[0.0], pos.values])) != 0).sum())
    return net, pos, n_changes


def metrics(r):
    eq = (1 + r).cumprod()
    cr = eq.iloc[-1] - 1
    years = len(r) / 252
    ar = (1 + cr) ** (1 / years) - 1 if years > 0 else np.nan
    ex = r - risk_free_daily(r.index).values
    sr = ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) > 0 else 0.0
    mdd = (1 - eq / eq.cummax().clip(lower=1.0)).max()
    return {"CR%": round(float(100 * cr), 2), "AR%": round(float(100 * ar), 2), "SR": round(float(sr), 2),
            "MDD%": round(100 * float(mdd), 2), "days": len(r)}


def buy_and_hold(ticker, decisions, cost):
    return simulate(ticker, [(decisions[0][0], "Buy")] + [(d, "Hold") for d, _ in decisions[1:]], cost)[0]


def random_agent(ticker, decisions, n_changes, cost, seeds=C.RANDOM_SEEDS):
    """Same dates, same number of position changes; each change goes to a random other state."""
    out = []
    n = len(decisions)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        k = min(n_changes, n)
        when = set(rng.choice(n, size=k, replace=False).tolist()) if k else set()
        cur, seq = 0, []
        for i, (d, _) in enumerate(decisions):
            if i in when:
                cur = rng.choice([p for p in (-1, 0, 1) if p != cur])
            seq.append((d, int(cur)))
        out.append(metrics(simulate(ticker, seq, cost)[0])["CR%"])
    return np.array(out)


def load_decisions(run_tag=None):
    rows = [json.loads(l) for l in open(DECISIONS)] if os.path.exists(DECISIONS) else []
    if run_tag:
        rows = [r for r in rows if r["run_tag"] == run_tag]
    cells = {}
    for r in rows:
        cells.setdefault((r["run_tag"], r["window"], r["ticker"]), {})[r["date"]] = r["rating"]   # last write wins
    out = {}
    for k, v in cells.items():
        seq = sorted(v.items())
        # A run cut short (spend cap, provider error) ends at its first ERROR: that date stays as the
        # endpoint, so the last good decision is held its full period, and nothing after it is scored.
        errs = [i for i, (_, r) in enumerate(seq) if r == "ERROR"]
        out[k] = seq[: errs[0] + 1] if errs else seq
    return out


def table(run_tag=None):
    res = []
    for (tag, window, t), dec in sorted(load_decisions(run_tag).items()):
        if len(dec) < 3:
            continue
        row = {"run_tag": tag, "window": window, "ticker": t, "decisions": len(dec),
               "ratings": pd.Series([r for _, r in dec]).value_counts().to_dict()}
        for cost in ("zero", C.MAIN_COST, "harsh"):
            net, pos, changes = simulate(t, dec, cost)
            row[f"ta_{cost}"] = metrics(net)
            row[f"bh_{cost}"] = metrics(buy_and_hold(t, dec, cost))
            row["position_changes"] = changes
        rnd = random_agent(t, dec, row["position_changes"], C.MAIN_COST)
        ta_cr = row[f"ta_{C.MAIN_COST}"]["CR%"]
        row["random_" + C.MAIN_COST] = {"median_CR%": round(float(np.median(rnd)), 2),
                                        "p5_CR%": round(float(np.percentile(rnd, 5)), 2),
                                        "p95_CR%": round(float(np.percentile(rnd, 95)), 2),
                                        "ta_percentile": round(float((rnd < ta_cr).mean() * 100), 1)}
        res.append(row)
    return res


if __name__ == "__main__":
    out = table()
    json.dump(out, open(os.path.join(C.RESULTS, "backtest.json"), "w"), indent=1)
    for r in out:
        print(f"{r['window']:8s} {r['ticker']:5s} n={r['decisions']:3d} TA {r['ta_' + C.MAIN_COST]}  "
              f"B&H {r['bh_' + C.MAIN_COST]}  random {r['random_' + C.MAIN_COST]}")
