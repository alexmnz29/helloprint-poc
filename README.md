# HelloPrint Quote Optimiser – PoC

This repository is a self-contained **proof-of-concept** that simulates HelloPrint’s home assigment solution for automated quotation:

* ⬆️ Upload a customer RFQ (PDF).  
* 🔍 Mock‐OCR extracts product, quantity, deadline.  
* 📝 Generates synthetic supplier offers that match the requested product.  
* 🤖 A pre-trained **XGBoost** model predicts win-probability (`p_win`) for every offer.  
* 📈 A linear optimiser picks the single offer that maximises `p_win × margin`
  while respecting a margin floor set in the sidebar.  
* 📊 Shows a ranked table, highlights the chosen supplier, and lets you
  download the full ranking.

Everything runs locally.

---

## 1. Quick start

```bash
# clone
git clone https://github.com/your-org/helloprint-quote-poc.git
cd helloprint-quote-poc
```

UV has been selected as package & project manager tool, please install it in order to launch the demo. 

Installation guide: https://docs.astral.sh/uv/getting-started/installation/

```bash
make env   # 1. Create the uv virtual-env and install deps
make data  # 2. Generate synthetic data
make train # 3. Train XGBoost model
make hpo   # (Optional) Run Optuna hyper-parameter optimization
make ui    # 4. Launch the Streamlit demo  (http://localhost:8501)
```

## 2. Makefile commands


```bash
make help        # list all available targets

make env         # create / refresh the uv virtual-env and install deps
make data        # regenerate synthetic CSVs in data/synthetic/
make train       # train XGBoost model → model.joblib
make hpo         # run Optuna hyper-parameter search  (default 30 trials)
make mlflow      # open the MLflow tracking UI at http://localhost:5000
make ui          # launch the Streamlit demo  (http://localhost:8501)
make test        # run unit tests with pytest
make clean       # delete generated CSVs, model.joblib and mlruns/
```

## 3. Folder Structure
.
├── src/
│   ├── synthetic_generator.py   # create data
│   ├── features.py              # joins + sklearn pipeline
│   ├── train.py                 # trains XGBoost, logs to MLflow, saves model.joblib
│   ├── inference.py             # scoring + LP optimiser
│   └── hpo.py                   # hyperparameter optimization with Optuna + logs to MLFlow
|   └── app.py                   # Streamlit UI
├── test/                        # fitted Pipeline (prep + XGBoost)
└── .streamlit/config.toml       # UI theme

## 4. Sample RFQ PDF

Any PDF will do.
The app mock-parses it into a dictionary like:

```bash
{
  "client_id": "C-123",
  "region": "NL",
  "product_type": "flyer",
  "quantity": 1500,
  "deadline": "2025-07-10"
}
```

# 5. Tech stack

| Layer                | Tools                         |
|-----------------------|-------------------------------|
| Orchestration (mock)  | pandas, faker                 |
| ML pipeline           | scikit-learn ColumnTransformer + XGBoost |
| Optimiser             | PuLP + CBC solver             |
| Experiment tracking   | MLflow (local file backend)   |
| UI                    | Streamlit + custom CSS        |
| Dependency manager    | uv (uv venv, uv sync)         |

## 6. Limitations & next steps
- OCR, supplier API calls, and PDF generation are mocked.
- Only one offer is selected.
- Model retraining is manual (run train.py); no CI-trigger yet.