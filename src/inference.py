# src/inference.py
"""
Inference + optimisation for the HelloPrint PoC.

Usage example
-------------
from inference import InferenceEngine
engine = InferenceEngine("model.joblib", margin_floor=0.20)

# df_offers has one row per supplier offer for the same deal -------------
# Required columns: identical to training set (unit_price, lead_time_days, ...)
best, ranked = engine.select_best_offer(df_offers)
print(best)          # => dict with supplier_id, p_win, margin_pct, utility
print(ranked.head()) # => DataFrame ordered by utility desc
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import pulp
import numpy as np


class InferenceEngine:
    def __init__(self, model_path: str | Path, margin_floor: float = 0.20) -> None:
        pipeline = joblib.load(model_path)
        self.prep  = pipeline.named_steps["prep"]    # ColumnTransformer
        self.model = pipeline.named_steps["clf"]     # XGBoost classifier
        self.margin_floor = margin_floor

    def _add_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the same engineered features used during training."""
        df = df.copy()
        df["price_delta_pct"] = df.unit_price / df.unit_price.min() - 1
        df["lead_delta_days"] = df.lead_time_days - df.lead_time_days.min()
        df["quantity_log"] = np.log1p(df.quantity)
        return df

    # ---------------------------- public -------------------------
    def predict_prob(self, df: pd.DataFrame) -> pd.Series:
        df_fe = self._add_derived(df)
        X = self.prep.transform(df_fe)
        return pd.Series(self.model.predict_proba(X)[:, 1], index=df.index)

    def select_best_offer(
        self, offers: pd.DataFrame
    ) -> Tuple[dict, pd.DataFrame]:
        """
        Score offers, optimise utility, and return:
        - best_offer (dict)
        - ranked_offers (DataFrame ordered by utility desc)
        """
        df = offers.copy().reset_index(drop=True)

        df = self._add_derived(df)
        
        df["p_win"] = self.predict_prob(df)
        df["utility"] = df["p_win"] * df["quoted_margin_pct"]

        # --- optimisation ---
        prob = pulp.LpProblem("choose_offer", pulp.LpMaximize)
        x_vars = pulp.LpVariable.dicts("x", df.index, 0, 1, cat="Binary")

        # objective
        prob += pulp.lpSum(df.utility[i] * x_vars[i] for i in df.index)

        # constraints
        prob += pulp.lpSum(df.quoted_margin_pct[i] * x_vars[i] for i in df.index) >= self.margin_floor
        prob += pulp.lpSum(x_vars[i] for i in df.index) == 1

        # solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError("No feasible offer meets margin floor")

        df["selected"] = [int(x_vars[i].value()) for i in df.index]
        df = df.sort_values("utility", ascending=False).reset_index(drop=True)
        best_row = df.loc[df.selected == 1].iloc[0]

        best_offer = {
            "supplier_id": best_row["supplier_id"],
            "p_win":       round(best_row["p_win"], 3),
            "margin_pct":  round(best_row["quoted_margin_pct"], 3),
            "utility":     round(best_row["utility"], 3),
        }
        return best_offer, df


# quick CLI test ----------------------------------------------------------
if __name__ == "__main__":
    import json, sys

    if len(sys.argv) != 2:
        print("Usage: python inference.py sample_offers.json")
        sys.exit(1)

    engine = InferenceEngine("model.joblib", margin_floor=0.20)
    offers_df = pd.read_json(sys.argv[1])
    best, ranked = engine.select_best_offer(offers_df)
    print("Selected offer →", best)
    print(ranked.head(5))