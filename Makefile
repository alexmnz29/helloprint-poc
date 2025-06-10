# ----------------- CONFIG -----------------
PY      := uv run --
DATA    := data/synthetic
MODEL   := model.joblib
TRIALS  ?= 30          # default Optuna trials

# ----------------- TARGETS ----------------
.PHONY: help env data train hpo ui test clean

help:      ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS=":.*?##"}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'

env:  ## Create/refresh .venv according to pyproject.toml
	uv venv
	uv sync

data:      ## Regenerate synthetic CSVs
	$(PY) python src/synthetic_generator.py

train:     ## Train model and save $(MODEL)
	$(PY) python src/train.py

hpo:       ## Hyper-parameter optimisation with Optuna (TRIALS=$(TRIALS))
	$(PY) python src/hpo.py --trials $(TRIALS)

mlflow: ## Launch MLFlow UI
	$(PY) mlflow ui

ui:        ## Launch Streamlit demo
	$(PY) streamlit run src/app.py

test:      ## Run unit tests
	$(PY) pytest -q

clean:     ## Remove synthetic data and model artefacts
	rm -rf $(DATA)/*.csv $(MODEL) mlruns