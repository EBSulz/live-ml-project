# Live Financial Market Prediction System

Real-time buy/hold/sell signal prediction using online machine learning (river), served via FastAPI, with full MLOps lifecycle on GCP.

## Architecture

```
Data Sources (Twelve Data / Alpha Vantage)
    │
    ▼
Feature Engineering (RSI, MACD, Bollinger, EMA, Volume)
    │
    ▼
Online Model (river – champion selected via MLflow experiment)
    │
    ▼
FastAPI (/predict, /health, /metrics)
    │
    ▼
Cloud Run (test / production with canary deployment)
```

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate live-ml-project
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your Twelve Data and/or Alpha Vantage keys
```

Or set them as environment variables:

```bash
export TWELVE_DATA_API_KEY=your_key_here
export ALPHA_VANTAGE_API_KEY=your_key_here
```

### 3. Run the experiment (train & compare models)

```bash
make experiment
# or:
python scripts/run_experiment.py \
    --experiment-name signal-classification-v1 \
    --symbol BTC/USD \
    --interval 1h \
    --outputsize 5000
```

This runs all 5 model families (HoeffdingTree, AdaptiveRandomForest, ADWINBagging, LogisticRegression, GaussianNB) across their hyperparameter grids, evaluates each via prequential (test-then-train) evaluation, and tags the best as "champion" in MLflow.

### 4. View results in MLflow

```bash
make mlflow-ui
# Open http://localhost:5000
```

### 5. Start the API

```bash
make serve
# Open http://localhost:8000/docs
```

### 6. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC/USD", "interval": "1h"}'
```

## Project Structure

```
├── src/
│   ├── config/          # Pydantic settings (env-driven)
│   ├── data_ingestion/  # Twelve Data + Alpha Vantage clients
│   ├── features/        # Incremental technical indicators
│   ├── models/          # Model zoo, experiment runner, champion wrapper
│   ├── validation/      # Data quality (GE) + drift detection (Deepchecks)
│   ├── api/             # FastAPI app with predict/health/metrics routes
│   └── observability/   # Cloud Logging + request middleware
├── tests/               # Pytest suite (>80% coverage target)
├── scripts/             # CLI tools: experiment, bootstrap, evaluate, promote
├── infra/               # Dockerfile, docker-compose, GCP setup
├── .github/workflows/   # CI (lint+test+model-gate) + GCP deploy pipeline
└── dvc.yaml             # Reproducible pipeline stages
```

## MLflow Experiment Lifecycle

### Running an experiment

The experiment runner (`src/models/experiment.py`) compares all model-hyperparameter combinations:

1. Creates an MLflow Experiment (e.g., `signal-classification-v1`)
2. For each model+hyperparam combo, creates a separate MLflow Run
3. Each Run uses prequential evaluation (predict, score, then learn)
4. Logs rolling metrics (learning curves), final metrics, confusion matrix, and model artifact
5. Auto-tags the best performing Run as `champion=true`

### Model Registry lifecycle

```
Experiment (many runs)
    │
    ▼ tag champion=true
MLflow Registry: Staging
    │
    ▼ CI quality gate (F1 > 0.60)
MLflow Registry: Production
    │
    ▼ promote_champion.py --vertex
Vertex AI Model Registry
```

## Promoting a Model from Test to Production

### Step 1: Run experiment and register the champion

```bash
python scripts/run_experiment.py \
    --experiment-name signal-classification-v1 \
    --register
```

This runs the full comparison, selects the champion, and registers it in MLflow Model Registry as `Staging`.

### Step 2: Verify in MLflow UI

Open `http://localhost:5000` and navigate to the Models tab. Confirm the new version is in `Staging`.

### Step 3: Promote to Production

```bash
python scripts/promote_champion.py \
    --experiment-name signal-classification-v1
```

This transitions the Staging model to Production in MLflow. The FastAPI app will load this version on next startup.

### Step 4: Deploy to GCP

Push to the `main` branch. The GitHub Actions pipeline will:

1. Build and push the Docker image to Artifact Registry
2. Wait for manual approval (GitHub Environment protection)
3. Deploy to Cloud Run with a 90/10 canary split
4. Auto-promote to 100% after health check passes

### Step 5 (optional): Register in Vertex AI

```bash
python scripts/promote_champion.py \
    --experiment-name signal-classification-v1 \
    --vertex
```

## GCP Setup

Run the one-time infrastructure setup:

```bash
export GCP_PROJECT_ID=your-project
export GITHUB_ORG=your-username
export GITHUB_REPO=your-repo
bash infra/gcp/setup.sh
```

This creates:
- Artifact Registry repository
- GCS buckets for model artifacts and DVC storage
- Secret Manager entries for API keys
- Service account with minimal IAM roles
- Workload Identity Federation for GitHub Actions (no JSON keys)

Then add the printed values as GitHub repository secrets:
- `GCP_PROJECT_ID`
- `GCP_SA_EMAIL`
- `WIF_PROVIDER`
- `TWELVE_DATA_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

## Development

```bash
make lint       # Black + Flake8
make format     # Auto-format with Black
make test       # Pytest with coverage
make docker-up  # Start app + MLflow via Docker Compose
```

## CI/CD Pipeline

| Workflow | Trigger | Jobs |
|---|---|---|
| `main.yml` | Push/PR | Lint, Test (coverage >80%), Model quality gate |
| `gcp-pipeline.yml` | PR → staging, Push main → production | Build, push to Artifact Registry, deploy to Cloud Run |

Production deployments require manual approval via GitHub Environments and use a canary traffic split (10% → 100%).
