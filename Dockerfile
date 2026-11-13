# tradingagents-after-the-cutoff: rerun every number and chart in a clean container.
#   docker build -t ta-cutoff .
#   docker run --rm -e OPENROUTER_API_KEY -e TA_MODEL -e TA_CUTOFF -e TA_PRICE_IN -e TA_PRICE_OUT \
#       -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" ta-cutoff
# Prices are not in the image (see data.py). For a local model, add --network host and TA_PROVIDER=ollama.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git make && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Makefile config.py data.py spend.py leak_probe.py run_agent.py backtest.py report.py selftest.py data.sha256 ./
ENV MPLCONFIGDIR=/tmp/mpl IN_DOCKER=1
CMD ["make", "reproduce"]
