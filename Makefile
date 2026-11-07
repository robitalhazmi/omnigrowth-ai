PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup data train test lint run clean-data

## setup: create the virtual environment and install pinned dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

## data: download public benchmark datasets and build the compressed Parquet marts
data:
	$(BIN)/python scripts/download_datasets.py
	$(BIN)/python scripts/build_data_mart.py
	du -sh data/processed/

## train: offline Prophet / AutoARIMA / Bass S-curve fitting (local laptop only)
train:
	$(BIN)/python scripts/train_forecasts.py

## test: run the full verification gate
test:
	$(BIN)/pytest tests/ -v

## lint: static analysis
lint:
	$(BIN)/ruff check src/ app/ tests/ scripts/

## run: launch the Streamlit app with production-like flags
run:
	$(BIN)/streamlit run app/main.py \
		--server.headless=true \
		--server.address=127.0.0.1 \
		--server.port=8501 \
		--server.maxUploadSize=5 \
		--browser.gatherUsageStats=false

## clean-data: remove raw downloads and processed artifacts
clean-data:
	rm -rf data/raw/* data/processed/*
	touch data/raw/.gitkeep data/processed/.gitkeep
