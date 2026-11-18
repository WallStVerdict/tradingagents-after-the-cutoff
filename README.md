# TradingAgents, tested on dates after its brain was trained

[![Watch the episode](https://img.youtube.com/vi/4suLfz-Dmk0/maxresdefault.jpg)](https://www.youtube.com/watch?v=4suLfz-Dmk0)

**[Watch the episode](https://www.youtube.com/watch?v=4suLfz-Dmk0)**

> Results from 11 Nov 2026. One run per window, one model, small samples: read it as a test, not a ranking. Education, not advice.

**Claim tested.** TradingAgents (TauricResearch, Apache-2.0) is a multi-agent LLM "trading firm":
analysts, a bull and a bear researcher, a trader, a risk team and a fund manager.
Its paper (Xiao, Sun, Luo, Wang, *TradingAgents: Multi-Agents LLM Financial Trading Framework*, arXiv 2412.20138, v7, 3 Jun 2025) reports, for **1 January to 29 March 2024**:

> "TradingAgents achieves at least a 23.21% cumulative return and a 24.90% annual return on the three sampled stocks, surpassing the best-performing baselines by a margin of 6.1%." (Section 6.1.1)

| Table 1 (paper) | Cumulative return | Annual return | Sharpe | Max drawdown | Buy and hold, same window |
|---|---|---|---|---|---|
| AAPL | 26.62% | 30.5% | 8.21 | 0.91% | -5.23% |
| GOOGL | 24.36% | 27.58% | 6.39 | 1.69% | 7.78% |
| AMZN | 23.21% | 24.90% | 5.60 | 2.11% | 17.1% |

The authors flag the Sharpe themselves: "The highest Sharpe Ratio exceeds our expected empirical range ... We believe the exceptionally high SR resulted from the phenomenon that there were few pullbacks in TradingAgents during that period" (footnote, p. 12).
The paper's models were gpt-4o-mini / gpt-4o for quick tasks and o1-preview for decisions (Section 4.3); no costs are mentioned.

The repo itself is more careful today: its README says backtest results "are not guaranteed to match any published figure" and calls the framework "a research scaffold ... not a strategy with a fixed, replicable return".

**Why this one.** It is the best-known open-source AI trading agent, the claim is dated and specific, and it raises a question every LLM backtest has: if the model was trained on text written after the test dates, it may already know how the trade ended.

## The test

1. **Leak probe.** No tools and no data in the prompt. We asked each model 312 questions. For six big US stocks (AAPL, GOOGL, AMZN, MSFT, NVDA, META) we asked for the close at every month end from Jan 2023 to Oct 2026, and whether the stock rose or fell that month. For the paper's three stocks we also asked about every week of the paper's window: did it close higher than the Friday before? We scored the answers against real prices and split them at each model's documented training cutoff:
   - **gpt-5-mini**: "May 31, 2024 knowledge cutoff" (OpenAI model page).
   - **Claude Haiku 5.5**: reliable knowledge and training-data cutoff Jun 2026 (Anthropic models overview). It ran through the Claude Code CLI in print mode, with no tools and no project context.
2. **The agent, unmodified.** TradingAgents at commit `9fa1196`, run on gpt-5-mini through fal.ai's OpenRouter route, on the paper's three stocks:
   - **Paper window:** 2 Jan to 28 Mar 2024. This is *before* gpt-5-mini's cutoff.
   - **Frozen holdout:** 1 Jul to 30 Sep 2026, *after* the cutoff. It ran once, with settings fixed before it started.
   - **Shake-down:** Apr 2026, two decisions on AAPL. Those numbers were never used.
   - **Cadence:** one decision every 10 trading days, 7 per stock per window. The paper decided daily. We measured about $0.07 a decision, so a $3 cap rules out a daily cadence.
   - **Settings:** reasoning effort "minimal", a 6,000-token output cap, and one debate round and one risk round, as in the repo's defaults.
3. **Fills and costs.** Buy or Overweight means long. Sell or Underweight means short. Hold keeps the current position. Every trade fills at the next day's open. Costs are 5 bps per side (spread plus slippage), plus 0.3% a year to borrow on short days. We also show 0 and 20 bps.
4. **Baselines.**
   - **Buy-and-hold** over the same days.
   - **A random agent:** on the same dates, it changes position as many times as TradingAgents did, in a random direction. We ran 1,000 seeds and report TradingAgents' percentile among them.
5. **Count the trials.** Every run is in `results/trials.jsonl` and every LLM call is in `results/ledger.jsonl`, including the ones we threw away.

What we changed around TradingAgents, and why:
- **Market analyst only.** TradingAgents' news, social and company-overview tools return today's feeds even for a past date (its README says so), which would hand the model the future. The paper used a historical news dataset that is not public.
- **Prices go through a point-in-time guard.** `yfinance.Ticker` is swapped for a local reader of the saved files. A request for any day after the analysis date gets nothing.
- **One cheap model on both tiers.** The paper used gpt-4o, gpt-4o-mini and o1-preview.

**Data:** Yahoo Finance daily bars via yfinance (TradingAgents' own default source, used for this episode only), split- and dividend-adjusted, pulled 11 Nov 2026. See `data/PULLED.txt` and `data.sha256`. SPY is included because TradingAgents uses it as the benchmark when it settles past decisions.

## Results

### Leak probe: does the model already know?

| | Questions | Right on up or down | Always guess the more common answer* | Close within 5% |
|---|---|---|---|---|
| gpt-5-mini, months before its cutoff (Jan 2023 to May 2024) | 102 | **67.6%** (p = 0.0002 vs a coin) | 73.5% | **41.2%** |
| gpt-5-mini, months after its cutoff (Jun 2024 to Oct 2026) | 174 | 55.2% (p = 0.10) | 56.3% | 6.3% |
| gpt-5-mini, paper window, week by week | 36 | 69.4% (p = 0.014) | 63.9% | 58.3% |
| Haiku 5.5, months before its cutoff (Jan 2023 to Jun 2026) | 252 | **81.7%** (p < 0.0001) | 62.7% | **82.9%** |
| Haiku 5.5, months after its cutoff (Jul to Oct 2026) | 24 | 50.0% | 62.5% | 16.7% |
| Haiku 5.5, paper window, week by week | 36 | 69.4% (p = 0.014) | 63.9% | 94.4% |

\* A hindsight benchmark: it already knows whether up or down was more common in that group.

Both models remember the paper's window.
Asked with no data at all, Haiku 5.5 named the close within 5% for 34 of 36 paper-window weeks, and gpt-5-mini for 21 of 36.
After each model's cutoff, price recall collapses (gpt-5-mini 41% to 6%; Haiku 83% to 17%) and direction falls to a coin flip.
gpt-5-mini's direction calls before its cutoff are well above chance but below simply guessing "up" in a rising market. Its stronger tell is price levels.
Charts: `results/leak_probe_*.png`.

### The agent, after costs (5 bps a side)

| Window | Stock | TradingAgents | Buy and hold | Random agent, median | TradingAgents vs random (percentile) | Paper, Table 1 |
|---|---|---|---|---|---|---|
| Paper, Jan to Mar 2024 (before the cutoff) | AAPL | -7.35% (Sharpe -1.60) | -7.05% | -0.34% | 6th | **+26.62%** (Sharpe 8.21) |
| | GOOGL | +9.68% (1.45) | +9.68% | 0.00% | 93rd | **+24.36%** (6.39) |
| | AMZN | +21.05% (2.96) | +21.05% | 0.00% | 93rd | **+23.21%** (5.60) |
| | **mean** | **+7.79%** | **+7.89%** | | | |
| Holdout, Jul to Sep 2026 (after the cutoff) | AAPL | -5.56% (-0.78) | +13.82% | -0.77% | 29th | |
| | GOOGL | -7.03% (-0.80) | -4.58% | -1.28% | 3rd | |
| | AMZN | +4.67% (0.65) | +4.67% | 0.00% | 91st | |
| | **mean** | **-2.64%** | **+4.64%** | | | |

- **Beat buy-and-hold:** 0 of 3 stocks in the paper window, 0 of 3 in the holdout.
- **Costs** barely matter here. The agent traded at most 3 times per stock, so 0 bps to 20 bps moves results by under 1 point.
- **AMZN and GOOGL in the paper window were buy-and-hold.** The agent's first call was Overweight and every later call was Hold or Overweight, so it bought on 3 Jan 2024 and never sold. Its "93rd percentile" against random agents just means being long in a quarter when the stock went up.
- **The holdout was cut short.** The fal.ai account ran out of balance after the 27 Aug decision (HTTP 403 "Exhausted balance"). The last decision is held to 11 Sep, the next scheduled date, and scored to there (50 trading days). The 11 and 25 Sep decisions were never made.
- **The same date can get a different call on a rerun.** 1 Apr 2026 came back Hold in one shake-down and Overweight in the next. Reasoning models take no temperature, so we could not pin it.
- **Our buy-and-hold differs from the paper's** (AAPL -7.05% vs -5.23%). We measure from the open after the first decision to the open after the last. The paper does not say which prices it used.

Charts: `results/equity_<window>_<stock>.png` (TradingAgents, buy-and-hold and the random agents' 5-95% band).

### Trials and spend
- **LLM runs:** 5 agent runs (one crashed on a missing benchmark file; one paper run was stopped and restarted cheaper after one call ran away to 65,536 output tokens) and 4 leak-probe runs (one pair thrown away after a bug in which month a question asked about).
- **Decisions:** 48. The table uses 42 of them: 21 in the paper window and 21 in the holdout, 6 of which errored.
- **gpt-5-mini via fal:** $2.18 in the ledger, of the $3 cap. 601 agent calls ($2.03) and 315 probe calls ($0.15).
- **Haiku 5.5 via `claude -p`:** 318 calls, $0, on a Claude subscription.

## Verdict: doesn't hold up

We could not get anywhere near the paper's +23% to +27% in a quarter.
On the same window, the same three stocks and the same agent code, a newer cheap model made about what buying and holding made (+7.8% vs +7.9%), and on two of the three stocks it simply was buy-and-hold.
On dates after the model's training cutoff it lost money (-2.6%) while buying and holding made +4.6%, and it beat buy-and-hold on none of the three stocks.
The leak probe shows why paper-window replays are risky today: both models we tried can recite the paper window's prices from memory, without any tools.

Caveats we can't remove on $3:
- one model, at minimal reasoning;
- the market analyst only;
- a decision every two weeks, not every day;
- one run per window;
- a holdout cut short to 50 trading days.

What would change our mind: a daily, all-analyst run on dates after the model's cutoff that beats buy-and-hold after costs, across many seeds and stocks.

## Reproduce

```
make selftest          # no LLM, no downloads: checks the simulator, the cost model, the date guard, the leak scorer
python data.py --yahoo   # prices as we used them (or --tiingo with a free TIINGO_API_KEY); then `python data.py --check`
export TA_PROVIDER=fal TA_MODEL=openai/gpt-5-mini TA_CUTOFF=2024-05 FAL_KEY=...   # or TA_PROVIDER=openrouter + OPENROUTER_API_KEY
# Haiku leak probe: TA_PROVIDER=claude_cli TA_MODEL=claude-haiku-5-5 TA_CUTOFF=2026-06 TA_REASONING= python leak_probe.py
make reproduce         # leak probe, paper window, holdout (once; needs HOLDOUT_GO=1), report + charts
make docker            # the same in python:3.12-slim
```
`SPEND_CAP_USD` (default 3) stops every LLM call once the ledger reaches it.
`make report` needs the run's `TA_*` settings exported (as above); without them it rebuilds `results.json` with empty per-stock results.

Code: MIT (see LICENSE). Third-party material: see NOTICE. Education, not advice.
