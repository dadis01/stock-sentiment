# Stock Market Sentiment Analysis Platform

Real-time sentiment analysis of financial news headlines for five major stocks,
backed by PostgreSQL, powered by FinBERT, and visualised through a Streamlit dashboard.

---

## 1. Project Overview

The platform has three independently runnable processes:

| Process | Entry point | Purpose |
|---------|-------------|---------|
| **Collector** | `data_collector/collector.py` | Fetches headlines from NewsAPI every 30 min, stores them in PostgreSQL, and queues article IDs for analysis |
| **Analyzer** | `data_analyzer/analyzer.py` (FastAPI) | Background worker drains the queue; REST endpoint scores headlines with FinBERT and writes results to DB |
| **Dashboard** | `web_app/app.py` (Streamlit) | Displays sentiment charts, headline tables, and live monitoring metrics |

Tracked tickers: **AAPL · TSLA · GOOGL · MSFT · AMZN**

---

## 2. Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                       DOCKER                                    │
 │   ┌─────────────────────────────────────────────────────────┐   │
 │   │  PostgreSQL 15  (localhost:5432 / stocksentiment DB)    │   │
 │   │   tables: articles · sentiments · prices               │   │
 │   └───────────────────────┬─────────────────────────────────┘   │
 └───────────────────────────│─────────────────────────────────────┘
                             │ psycopg2
          ┌──────────────────┼──────────────────┐
          │                  │                  │
  ┌───────▼──────┐  ┌────────▼────────┐  ┌─────▼────────────┐
  │  Collector   │  │    Analyzer     │  │   Dashboard      │
  │  (schedule)  │  │  (FastAPI +     │  │  (Streamlit)     │
  │              │  │   uvicorn)      │  │                  │
  │  NewsAPI ──► │  │                 │  │  Plotly charts   │
  │  yfinance ──►│  │ FinBERT model   │  │  Headlines table │
  │              │  │ yfinance        │  │  Metrics panel   │
  │  queue.Queue─┼─►│ background      │  │                  │
  └──────────────┘  │ thread          │  └──────────────────┘
                    └─────────────────┘
                        REST API
                    GET  /health
                    POST /analyze
```

---

## 3. Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | `python --version` |
| Docker Desktop | latest | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) |
| DBeaver (optional) | latest | GUI SQL client — [dbeaver.io](https://dbeaver.io) |

---

## 4. Setup Steps

### a. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-sentiment.git
cd stock-sentiment
```

### b. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and replace `your_newsapi_key_here` with your real key from
[newsapi.org](https://newsapi.org) (free tier supports 100 requests/day).

```
DATABASE_URL=postgresql://admin:secret@localhost:5432/stocksentiment
NEWSAPI_KEY=<your key>
ANALYZER_URL=http://localhost:8000
```

### c. Start PostgreSQL in Docker

```bash
docker-compose up -d
```

Verify it's running:

```bash
docker ps
```

### d. Install Python dependencies

```bash
pip install -r requirements.txt
```

> On first run, `transformers` will download the FinBERT model (~500 MB) from Hugging Face.

### e. Initialise the database schema

```bash
python -c "from database.db import init_db; init_db()"
```

---

## 5. Running All Three Processes

Open **three separate terminal windows** in the project root:

**Terminal 1 — Analyzer (FastAPI)**
```bash
uvicorn data_analyzer.analyzer:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Collector**
```bash
python data_collector/collector.py
```

**Terminal 3 — Dashboard (Streamlit)**
```bash
streamlit run web_app/app.py
```

The dashboard opens automatically at [http://localhost:8501](http://localhost:8501).

---

## 6. Connecting DBeaver to the Local Database

1. Open DBeaver → **New Database Connection** → choose **PostgreSQL**
2. Fill in the connection details:

| Field | Value |
|-------|-------|
| Host | `localhost` |
| Port | `5432` |
| Database | `stocksentiment` |
| Username | `admin` |
| Password | `secret` |

3. Click **Test Connection**, then **Finish**.
4. Browse `stocksentiment > Schemas > public > Tables` to inspect `articles`, `sentiments`, and `prices`.

---

## 7. Running Tests

```bash
pytest tests/ -v
```

All tests mock external APIs (NewsAPI, FinBERT, yfinance) and use an in-memory SQLite
database, so **no Docker and no API keys are required** to run the test suite.

---

## 8. Deploying to Heroku

### Prerequisites
- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed and logged in
- A Heroku account

### Steps

```bash
# Create the app
heroku create your-app-name

# Add Heroku Postgres (free Eco tier)
heroku addons:create heroku-postgresql:essential-0

# Set environment variables
heroku config:set NEWSAPI_KEY=your_newsapi_key_here
heroku config:set ANALYZER_URL=https://your-app-name.herokuapp.com

# Heroku sets DATABASE_URL automatically from the add-on

# Deploy
git push heroku main

# Initialise the schema on Heroku Postgres
heroku run python -c "from database.db import init_db; init_db()"

# Scale dynos (one per Procfile entry)
heroku ps:scale web=1 collector=1 analyzer=1

# Open the dashboard
heroku open
```

> **Note:** The free tier only supports one dyno. For production use, scale as needed and
> consider a paid Postgres plan for connection limits.

---

## 9. Free API Links

| Service | URL | Notes |
|---------|-----|-------|
| NewsAPI | [newsapi.org](https://newsapi.org) | Free tier: 100 req/day, register for API key |
| FinBERT (Hugging Face) | [huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | Free, downloaded automatically by `transformers` |

---

## Project Structure

```
stock-sentiment/
├── database/
│   └── db.py                  # psycopg2 DB layer (get_connection, init_db, CRUD)
├── data_collector/
│   └── collector.py           # NewsAPI + yfinance fetcher, schedule loop
├── data_analyzer/
│   ├── analyzer.py            # FastAPI app + background queue worker
│   └── sentiment.py           # FinBERT wrapper (load_finbert, analyze)
├── web_app/
│   └── app.py                 # Streamlit dashboard (4 sections)
├── tests/
│   ├── test_db.py             # DB unit tests (SQLite shim)
│   ├── test_collector.py      # Collector unit tests (mocked)
│   ├── test_analyzer.py       # Analyzer unit tests (mocked + TestClient)
│   └── test_integration.py    # End-to-end pipeline tests
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI (postgres service + pytest)
├── docker-compose.yml         # PostgreSQL 15 service
├── .env.example               # Environment variable template
├── Procfile                   # Heroku process definitions
├── requirements.txt           # Pinned Python dependencies
└── README.md
```
