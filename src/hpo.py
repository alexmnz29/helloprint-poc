# src/hpo.py
"""
Hyper-parameter optimisation for the HelloPrint MVP.
----------------------------------------------------
* Uses Optuna (TPE sampler) to tune XGBoost params.
* Logs every trial to MLflow via the MLflowCallback.
* Persists the best Pipeline as model.joblib.

Run:
    uv pip install optuna mlflow
    uv shell
    python src/hpo.py --trials 30
"""

from __future__ import annotations
import argparse
import joblib
import optuna
from optuna.integration.mlflow import MLflowCallback

import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from features import load_raw, build_preprocess, prepare_xy


# ------------------------------------------------------------------------
# prepare data once (shared across trials)
# ------------------------------------------------------------------------
_df = load_raw()
_X, _y = prepare_xy(_df)
X_tr, X_val, y_tr, y_val = train_test_split(
    _X, _y, test_size=0.20, random_state=42, stratify=_y
)


# ------------------------------------------------------------------------
# objective function
# ------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 400),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("prep", build_preprocess()),
            ("clf", xgb.XGBClassifier(**params)),
        ]
    )

    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)

    # Save the pipeline for the best trial later
    trial.set_user_attr("pipeline", pipe)

    return auc


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------
def main(n_trials: int = 30) -> None:
    mlcb = MLflowCallback(metric_name="val_auc")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[mlcb])

    best = study.best_trial
    print(f"Best AUC={best.value:.3f}  params={best.params}")

    # retrieve the fitted Pipeline from the best trial and save it
    best_pipe = best.user_attrs["pipeline"]
    joblib.dump(best_pipe, "model.joblib")
    print("✓ Best model saved to model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default 30)")
    args = parser.parse_args()
    main(args.trials)