"""Every LLM call goes through here: priced, logged to a ledger, and stopped at a hard cap.

Doors:
- `chat()` for the leak probe: an OpenAI-compatible HTTP call (fal's OpenRouter route, OpenRouter,
  OpenAI, Ollama ...), or the Claude Code CLI in print mode (`claude_cli`, no tools, no project context).
- `SpendGuard`, a LangChain callback handed to TradingAgents so its own calls are counted the same way.
The cap is SPEND_CAP_USD for the whole study (all ledger rows), never more than the QS_BUDGET that
scripts/paid.sh exports. Claude CLI calls cost $0 on a subscription but have their own call cap.
"""
import datetime
import json
import os
import subprocess
import tempfile
import time

import requests
from langchain_core.callbacks import BaseCallbackHandler

import config as C

BASE_URLS = {
    "fal": "https://fal.run/openrouter/router/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}
KEY_ENV = {"fal": "FAL_KEY", "openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY", "ollama": None}
# Fallback list prices, USD per 1M tokens (in, out), used when the response carries no cost.
LIST_PRICE = {"openai/gpt-5-mini": (0.25, 2.00), "openai/gpt-4o-mini": (0.15, 0.60)}


class OverBudget(RuntimeError):
    pass


def _rows():
    if not os.path.exists(C.LEDGER):
        return []
    out = []
    for line in open(C.LEDGER):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def spent():
    return sum(float(r.get("usd") or 0) for r in _rows())


def cli_calls():
    return sum(int(r.get("calls") or 0) for r in _rows() if r.get("provider") == "claude_cli")


def cap():
    c = C.SPEND_CAP_USD
    if os.environ.get("QS_BUDGET"):
        c = min(c, spent() + float(os.environ["QS_BUDGET"]))
    return c


def check(provider=None):
    if (provider or C.PROVIDER) == "claude_cli":
        if cli_calls() >= C.CLAUDE_CLI_CALL_CAP:
            raise OverBudget(f"Claude CLI call cap reached: {cli_calls()} of {C.CLAUDE_CLI_CALL_CAP}")
        return
    if spent() >= cap():
        raise OverBudget(f"spend cap reached: ${spent():.4f} of ${cap():.2f}")


def price(model, tin, tout, reported=None):
    p_in, p_out = LIST_PRICE.get(model, (C.PRICE_IN_PER_M, C.PRICE_OUT_PER_M))
    listed = tin / 1e6 * p_in + tout / 1e6 * p_out
    return max(listed, float(reported or 0))   # the higher of the two, to stay on the safe side of the cap


def log(what, tokens_in, tokens_out, usd, provider=None, model=None, extra=None):
    os.makedirs(os.path.dirname(C.LEDGER), exist_ok=True)
    row = {"t": datetime.datetime.now().isoformat(timespec="seconds"), "usd": round(usd, 6), "what": what,
           "provider": provider or C.PROVIDER, "model": model or C.MODEL, "in": tokens_in, "out": tokens_out,
           "calls": 1, **(extra or {})}
    with open(C.LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    return usd


def _http_chat(messages, what, max_tokens, retries=4):
    url = (os.environ.get("TA_BASE_URL") or BASE_URLS[C.PROVIDER]).rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key_env = KEY_ENV.get(C.PROVIDER)
    if key_env:
        headers["Authorization"] = "Bearer " + os.environ[key_env]   # never printed
    body = {"model": C.MODEL, "messages": messages}
    if C.REASONING_EFFORT:
        body["reasoning"] = {"effort": C.REASONING_EFFORT}   # reasoning models ignore temperature
    else:
        body.update({"temperature": C.TEMPERATURE, "max_tokens": max_tokens})
    for i in range(retries):
        r = requests.post(url, headers=headers, json=body, timeout=180)
        if r.status_code == 200:
            j = r.json()
            u = j.get("usage") or {}
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            log(what, tin, tout, price(C.MODEL, tin, tout, u.get("cost")))
            return j["choices"][0]["message"]["content"] or ""
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (i + 1))
            continue
        raise RuntimeError(f"LLM call failed: HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("LLM call failed after retries")


def _claude_cli_chat(messages, what):
    """Claude Code CLI in print mode: run from an empty temp dir, no tools, no settings, no MCP, no session."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    msg = ""
    with tempfile.TemporaryDirectory(prefix="probe_") as empty:
        cmd = ["claude", "-p", "--model", C.MODEL, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--disable-slash-commands", "--no-session-persistence", "--system-prompt", system,
               "--output-format", "json", user]
        for i in range(3):
            p = subprocess.run(cmd, cwd=empty, capture_output=True, text=True, timeout=300)
            try:
                j = json.loads(p.stdout)
            except Exception:
                j = {}
            if p.returncode == 0 and not j.get("is_error"):
                u = j.get("usage") or {}
                tin = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                log(what, tin, u.get("output_tokens", 0), 0.0, extra={
                    "api_equiv_usd": j.get("total_cost_usd"), "resolved_model": ",".join((j.get("modelUsage") or {}).keys())})
                return j.get("result") or ""
            msg = (j.get("result") or p.stderr or p.stdout or "")[:200]
            if "limit" in msg.lower():
                raise OverBudget(f"Claude CLI usage limit: {msg}")
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"claude -p failed: {msg}")


def chat(messages, what, max_tokens=200):
    """One chat completion with no tools. Returns the text."""
    check()
    if C.PROVIDER == "claude_cli":
        return _claude_cli_chat(messages, what)
    return _http_chat(messages, what, max_tokens)


class SpendGuard(BaseCallbackHandler):
    """Counts TradingAgents' own LLM calls and stops the run at the cap."""

    raise_error = True   # LangChain swallows callback errors unless this is set

    def __init__(self, what):
        self.what = what
        self.calls = 0

    def on_llm_start(self, *a, **k):
        check()

    def on_chat_model_start(self, *a, **k):
        check()

    def on_llm_end(self, response, **k):
        lo = response.llm_output or {}
        u = lo.get("token_usage") or lo.get("usage") or {}
        tin = u.get("prompt_tokens", 0) or u.get("input_tokens", 0)
        tout = u.get("completion_tokens", 0) or u.get("output_tokens", 0)
        if not (tin or tout):
            for gens in response.generations:
                for g in gens:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                    tin += um.get("input_tokens", 0)
                    tout += um.get("output_tokens", 0)
        self.calls += 1
        log(self.what, tin, tout, price(C.MODEL, tin, tout, u.get("cost")))
