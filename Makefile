.PHONY: install lint format test serve experiment promote bootstrap clean

install:
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n live-ml-project pip install -e ".[dev]"

lint:
	black --check src/ tests/ scripts/
	flake8 src/ tests/ scripts/

format:
	black src/ tests/ scripts/

test:
	pytest --cov=src --cov-fail-under=80

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

experiment:
	python scripts/run_experiment.py \
		--experiment-name signal-classification-v1 \
		--symbol BTC/USD \
		--interval 1h \
		--outputsize 5000

promote:
	python scripts/promote_champion.py \
		--experiment-name signal-classification-v1

bootstrap:
	python scripts/bootstrap_model.py \
		--experiment-name signal-classification-v1 \
		--symbol BTC/USD \
		--interval 1h

evaluate:
	python scripts/evaluate_model.py \
		--experiment-name signal-classification-v1

mlflow-ui:
	mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

docker-build:
	docker build -t market-predictor:latest -f infra/Dockerfile .

docker-up:
	docker compose -f infra/docker-compose.yml up -d

docker-down:
	docker compose -f infra/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage build dist *.egg-info
