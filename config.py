"""Every setting for the study in one place. Change nothing here after the holdout is run.

The holdout window is sealed: `run_agent.py --window holdout` refuses to run twice, and the
config hash it ran with is stored in results/trials.jsonl.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))

# The paper's three stocks (TradingAgents v7, arXiv 2412.20138, Table 1).
TICKERS = ["AAPL", "GOOGL", "AMZN"]
NAMES = {"AAPL": "Apple", "GOOGL": "Alphabet (Google) class A", "AMZN": "Amazon",
         "MSFT": "Microsoft", "NVDA": "Nvidia", "META": "Meta Platforms"}

# Test windows (first and last analysis date, inclusive).
WINDOWS = {
    # The paper's window: "January 1st to March 29th, 2024" (Section 5.1).
    "paper": ("2024-01-02", "2024-03-28"),
    # Shake-down window: used to debug the harness. Results are reported but not the verdict.
    "dev": ("2026-04-01", "2026-06-30"),
    # Frozen holdout: run once, with the final config, after everything else is fixed.
    "holdout": ("2026-07-01", "2026-09-30"),
}

# One decision every N trading days. The paper decided daily; at ~$0.055 a decision on gpt-5-mini (measured in
# the shake-down), every 10 trading days is what fits both windows under the $3 cap.
EVERY_N_TRADING_DAYS = int(os.environ.get("EVERY_N", "10"))

# The model. Any OpenAI-compatible endpoint works (OpenRouter, Ollama, OpenAI ...).
# KNOWLEDGE_CUTOFF is the model maker's documented training cutoff; the leak probe measures it too.
# Providers: "fal" (fal.ai's OpenAI-compatible OpenRouter route, FAL_KEY), "openrouter", "openai", "ollama",
# and "claude_cli" (the Claude Code CLI in print mode, on a Claude subscription; leak probe only).
PROVIDER = os.environ.get("TA_PROVIDER", "fal")
MODEL = os.environ.get("TA_MODEL", "")
KNOWLEDGE_CUTOFF = os.environ.get("TA_CUTOFF", "")          # YYYY-MM, from the model card
PRICE_IN_PER_M = float(os.environ.get("TA_PRICE_IN", "0"))   # USD per 1M input tokens
PRICE_OUT_PER_M = float(os.environ.get("TA_PRICE_OUT", "0")) # USD per 1M output tokens

# Reasoning effort for reasoning models (gpt-5-mini): "minimal", "low", "medium", "high"; "" = provider default.
REASONING_EFFORT = os.environ.get("TA_REASONING", "minimal")
# Output cap per call (reasoning included). One uncapped call ran away to 65,536 tokens ($0.13) on 11 Nov.
MAX_TOKENS = int(os.environ.get("TA_MAX_TOKENS", "6000"))
# The Claude CLI runs on a subscription with usage limits: a hard cap on calls, logged at $0.
CLAUDE_CLI_CALL_CAP = int(os.environ.get("CLAUDE_CLI_CALL_CAP", "400"))

# Hard spend cap for every LLM call in this study (agent runs + leak probe together).
SPEND_CAP_USD = float(os.environ.get("SPEND_CAP_USD", "3.0"))
LEDGER = os.environ.get("LEDGER", os.path.join(RESULTS, "ledger.jsonl"))

# TradingAgents settings: the cheapest faithful graph. Market analyst only (price + indicators):
# its news, social and fundamentals tools read today's feeds, not the feeds as of the trade date.
ANALYSTS = ("market",)
DEBATE_ROUNDS = 1
RISK_ROUNDS = 1
TEMPERATURE = 0.0

# Costs, the same presets in every episode (basis points per unit of position traded).
COSTS = {
    "zero": {"bps": 0.0, "borrow_pa": 0.0},
    "retail": {"bps": 5.0, "borrow_pa": 0.003},   # 2 bps half-spread + 3 bps slippage; 0.3%/yr to borrow for a short
    "harsh": {"bps": 20.0, "borrow_pa": 0.01},
}
MAIN_COST = "retail"

RANDOM_SEEDS = 1000

# Leak probe: which stocks, which months.
PROBE_TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT", "NVDA", "META"]
PROBE_START = "2023-01"
PROBE_END = "2026-10"


def config_hash():
    blob = json.dumps({k: v for k, v in globals().items()
                       if k.isupper() and k not in ("LEDGER", "HERE", "DATA", "RESULTS")}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
