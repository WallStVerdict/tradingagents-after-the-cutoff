"""Collect everything into results/results.json and draw the charts.

  python report.py
Charts: equity per window and stock (TradingAgents vs buy-and-hold vs the random agents' 5-95% band),
and the leak-probe hit rate before and after the model's cutoff.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import backtest as B
import config as C
from spend import spent

TEAL, GOLD, GREY, INK = "#2ec4b6", "#f4b942", "#8a94a6", "#0f1b2d"


def trials():
    p = os.path.join(C.RESULTS, "trials.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pooled(rows):
    out = {}
    for w in sorted({r["window"] for r in rows}):
        rs = [r for r in rows if r["window"] == w]
        k = "ta_" + C.MAIN_COST
        out[w] = {"stocks": len(rs),
                  "ta_mean_CR%": round(float(np.mean([r[k]["CR%"] for r in rs])), 2),
                  "ta_mean_SR": round(float(np.mean([r[k]["SR"] for r in rs])), 2),
                  "bh_mean_CR%": round(float(np.mean([r["bh_" + C.MAIN_COST]["CR%"] for r in rs])), 2),
                  "random_median_CR%": round(float(np.mean([r["random_" + C.MAIN_COST]["median_CR%"] for r in rs])), 2),
                  "ta_mean_percentile_vs_random": round(float(np.mean([r["random_" + C.MAIN_COST]["ta_percentile"] for r in rs])), 1),
                  "beat_bh": sum(r[k]["CR%"] > r["bh_" + C.MAIN_COST]["CR%"] for r in rs)}
    return out


def equity_chart(row):
    dec = B.load_decisions(row["run_tag"])[(row["run_tag"], row["window"], row["ticker"])]
    net, pos, ch = B.simulate(row["ticker"], dec, C.MAIN_COST)
    bh = B.buy_and_hold(row["ticker"], dec, C.MAIN_COST)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)
    bands = []
    for s in range(200):
        rng = np.random.default_rng(s)
        n = len(dec)
        when = set(rng.choice(n, size=min(ch, n), replace=False).tolist())
        cur, seq = 0, []
        for i, (d, _) in enumerate(dec):
            if i in when:
                cur = int(rng.choice([p for p in (-1, 0, 1) if p != cur]))
            seq.append((d, cur))
        bands.append((1 + B.simulate(row["ticker"], seq, C.MAIN_COST)[0]).cumprod().values)
    bands = np.array(bands)
    x = net.index
    ax.fill_between(x, np.percentile(bands, 5, axis=0), np.percentile(bands, 95, axis=0), color=GREY, alpha=0.25,
                    label="random agent, same trade count (5-95%)")
    ax.plot(x, (1 + bh).cumprod(), color=GOLD, lw=2, label="buy and hold")
    ax.plot(x, (1 + net).cumprod(), color=TEAL, lw=2.6, label="TradingAgents, after costs")
    ax.axhline(1, color=GREY, lw=0.6)
    ax.set_title(f"{row['ticker']}, {row['window']} window ({x[0]:%d %b %Y} to {x[-1]:%d %b %Y})", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.legend(facecolor=INK, labelcolor="white", frameon=False, loc="upper left")
    p = os.path.join(C.RESULTS, f"equity_{row['window']}_{row['ticker']}.png")
    fig.savefig(p, bbox_inches="tight", facecolor=INK)
    plt.close(fig)
    return p


def leak_chart(lp):
    g = lp.get("by_quarter", {})
    if not g:
        return None
    qs = list(g)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)
    cut = lp.get("documented_cutoff") or "9999-99"
    cols = [TEAL if q[:4] + "-" + f"{(int(q[-1]) - 1) * 3 + 1:02d}" <= cut else GOLD for q in qs]
    ax.bar(qs, [g[q]["direction_hit_rate"] * 100 for q in qs], color=cols)
    ax.axhline(50, color="white", lw=1, ls="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("right about up or down (%)", color="white")
    ax.set_title(f"Does {lp['model']} already know? (cutoff {cut}; teal = before, gold = after)", color="white")
    ax.tick_params(colors="white", labelrotation=45)
    p = os.path.join(C.RESULTS, "leak_probe_" + lp["model"].replace("/", "_") + ".png")
    fig.savefig(p, bbox_inches="tight", facecolor=INK)
    plt.close(fig)
    return p


def main():
    run_tag = f"{C.MODEL}|{C.config_hash()}"   # only the final configuration's decisions
    rows = [x for x in B.table(run_tag) if x["window"] != "dev"]
    import glob
    lps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(C.RESULTS, "leak_probe_*.json")))
           if not p.endswith("answers.jsonl")]
    tr = trials()
    out = {"model": C.MODEL, "provider": C.PROVIDER, "documented_cutoff": C.KNOWLEDGE_CUTOFF,
           "config_hash": C.config_hash(), "run_tag": run_tag,
           "settings": {"reasoning_effort": C.REASONING_EFFORT, "max_tokens": C.MAX_TOKENS, "analysts": C.ANALYSTS,
                        "debate_rounds": C.DEBATE_ROUNDS, "risk_rounds": C.RISK_ROUNDS}, "costs_main": C.COSTS[C.MAIN_COST], "windows": C.WINDOWS,
           "every_n_trading_days": C.EVERY_N_TRADING_DAYS,
           "trials": {"agent_runs": len(tr), "decisions": sum(t.get("decisions", 0) for t in tr),
                      "holdout_runs": sum(t["window"] == "holdout" for t in tr)},
           "spend_usd": round(spent(), 4), "pooled": pooled(rows) if rows else {}, "per_stock": rows,
           "leak_probe": {lp["model"]: lp for lp in lps}}
    json.dump(out, open(os.path.join(C.RESULTS, "results.json"), "w"), indent=1)
    for r in rows:
        equity_chart(r)
    for lp in lps:
        leak_chart(lp)
    print(json.dumps(out["pooled"], indent=1))
    print("trials:", out["trials"], "spend: $", out["spend_usd"])


if __name__ == "__main__":
    main()
