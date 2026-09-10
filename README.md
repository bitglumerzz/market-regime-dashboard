# Market Regime & Elliott Wave Forecasting Dashboard

A Streamlit dashboard that combines classical Elliott Wave analysis, an HMM-based
market regime detector, and an LLM (Claude) as a scoring/rescoring layer over
competing wave-count hypotheses — pulling live chart context from TradingView
through an MCP bridge.

## What it does

- **Wave classification** (`elliott.py`, `wave_sequencer.py`, `wave_validator.py`,
  `zigzag.py`): detects swing structure on OHLC data and scores competing Elliott
  Wave counts against each other, rather than committing to a single "the" count.
- **Regime detection** (`hmm_model.py`, `regime_labeler.py`): a Hidden Markov
  Model over price/volatility features to label the current market regime
  (trending / ranging / volatile) as context for the wave read.
- **LLM as a judge, not an oracle** (`claude_rescore.py`, `claude_call_log.py`):
  Claude re-scores the model's own wave-count candidates using the same
  structured criteria a human technician would — confidence is calibrated
  against realized outcomes over time (`prediction_log.py`), not taken at
  face value from a single call.
- **TradingView bridge** (`tv_bridge.py`, `tv_bridge_client.py`,
  `tv_mcp_analysis_store.py`): a local service that talks to TradingView via
  MCP to pull live chart state, annotate it, and screenshot it
  (`tv_screenshot.py`, `tv_annotate.py`) for the LLM to reason over.
- **Backtesting & sensitivity** (`backtest.py`, `monte_carlo.py`,
  `sensitivity.py`, `stress_tests.py`): validates the pipeline against
  historical data rather than trusting the live forecast alone.
- **Multi-timeframe / multi-asset** (`multi_tf.py`, `multi_asset.py`,
  `portfolio.py`): the same pipeline applied across timeframes and a
  watchlist, not a single chart.
- Sentiment and influencer-tracking modules exist (`sentiment.py`,
  `twitter_sentiment.py`, `influencer_*`) as auxiliary signal sources; the
  scraped/accumulated data they produce is not included in this repo (see
  below).

## What's not in this repo

Everything here is the pipeline itself. Deliberately excluded (see
`.gitignore`): accumulated runtime state and prediction history
(`predictions/`, `wave_journal/`, `tv_ground_truth/`), scraped third-party
data (`twitter_sentiment/`, `tv_ideas/`), a knowledge base distilled from a
paid third-party trading course (not mine to redistribute), API logs
(`claude_calls/`), and all `.env`/log files. None of that is needed to read
or run the code — it's just excluded because it's either regenerated at
runtime, third-party sourced, or simply not meant for a public repo.

## Stack

Python, Streamlit, `hmmlearn`, `pandas`/`numpy`, `anthropic` SDK, `ccxt` /
`yfinance` for market data, Plotly for charts, Docker for deployment.

## Setup

```sh
cp .env.example .env   # fill in ANTHROPIC_API_KEY at minimum
docker compose up -d --build
# or locally: pip install -r requirements.txt && streamlit run app.py
```

`tv_bridge.py` runs as a separate local service (see
`com.example.tv_bridge.plist` for a macOS launchd template) — it's what
gives the LLM live TradingView chart context via MCP.
