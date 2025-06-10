# src/train.py
"""
Train XGBoost on synthetic history, log everything to MLflow,
and save the fitted Pipeline as model.joblib.
Run with:  uv shell && python src/train.py
"""

import joblib
import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import load_raw, build_preprocess, prepare_xy

# ----------------------------------------------------------------------
# 1. Load data and split
# ----------------------------------------------------------------------
df_raw = load_raw()
X, y = prepare_xy(df_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 2. Build pipeline
# ----------------------------------------------------------------------
xgb_params = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("prep", build_preprocess()),
        ("clf", xgb.XGBClassifier(**xgb_params)),
    ]
)

# ----------------------------------------------------------------------
# 3. Train + MLflow logging
# ----------------------------------------------------------------------
with mlflow.start_run(run_name="xgb_mvp") as run:
    # log hyper-parameters
    mlflow.log_params(xgb_params)

    # fit
    pipeline.fit(X_train, y_train)

    # validation metric
    proba_val = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba_val)
    mlflow.log_metric("val_auc", auc)
    print(f"Validation AUC: {auc:.3f}")

    # save artefact locally
    joblib.dump(pipeline, "model.joblib")
    print("✓ model.joblib saved")

    # log artefact to MLflow
    mlflow.log_artifact("model.joblib")

    print(f"MLflow run_id: {run.info.run_id}")