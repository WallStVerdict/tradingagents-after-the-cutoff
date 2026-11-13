# `make reproduce` builds a local venv on first use, then: leak probe, the agent on every window, the report.
# The LLM settings come from the environment (TA_PROVIDER, TA_MODEL, TA_CUTOFF, TA_PRICE_IN, TA_PRICE_OUT,
# and the provider's API key). SPEND_CAP_USD (default 3) stops every LLM call once the ledger reaches it.
VENV   ?= .venv
DOCKER ?= docker

ifdef IN_DOCKER
PY    := python
STAMP :=
else
PY    := $(VENV)/bin/python
STAMP := $(VENV)/.installed
endif

.PHONY: reproduce selftest probe paper holdout report check docker clean

reproduce: $(STAMP) check probe paper holdout report

check: $(STAMP)
	@$(PY) data.py --check

probe: $(STAMP)
	@$(PY) leak_probe.py

paper: $(STAMP)
	@$(PY) run_agent.py --window paper --purpose "paper window"

holdout: $(STAMP)
	@$(PY) run_agent.py --window holdout --purpose "frozen holdout, final config"

report: $(STAMP)
	@$(PY) report.py

# No LLM, no data download: synthetic prices and fake ratings, to check the plumbing.
selftest: $(STAMP)
	@$(PY) selftest.py

$(VENV)/.installed: requirements.txt
	uv venv -q -p 3.12 $(VENV) 2>/dev/null || python3 -m venv $(VENV)
	uv pip install -q -p $(VENV)/bin/python -r requirements.txt 2>/dev/null || $(VENV)/bin/pip install -q -r requirements.txt
	@touch $@

docker:
	$(DOCKER) build -t ta-cutoff .
	$(DOCKER) run --rm -e OPENROUTER_API_KEY -e TA_PROVIDER -e TA_MODEL -e TA_CUTOFF -e TA_PRICE_IN -e TA_PRICE_OUT \
	    -v "$$PWD/data:/app/data" -v "$$PWD/results:/app/results" ta-cutoff

clean:
	rm -rf results
