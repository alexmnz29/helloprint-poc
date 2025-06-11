# src/features.py
"""
Join raw synthetic tables and build a scikit-learn pipeline
that transforms them into a feature matrix suitable for XGBoost.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data/synthetic")


# ----------------------------------------------------------------------
# 1. Load & join raw tables
# ----------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Return one row per supplier offer with deal & supplier context."""
    deals     = pd.read_csv(DATA_DIR / "deals.csv")
    offers    = pd.read_csv(DATA_DIR / "supplier_offers.csv")
    outcome   = pd.read_csv(DATA_DIR / "deal_outcome.csv")
    suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")

    df = (
        offers
        .merge(deals,     on="deal_id",   how="left", suffixes=("", "_deal"))
        .merge(suppliers, on="supplier_id", how="left", suffixes=("", "_sup"))
        .merge(outcome,   on="deal_id", how="left")
    )
    return df


# ----------------------------------------------------------------------
# 2. Feature-engineering helpers
# ----------------------------------------------------------------------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price_delta_pct & lead_delta_days within each deal."""
    df = df.copy()

    df["price_delta_pct"] = (
        df.unit_price
        / df.groupby("deal_id").unit_price.transform("min")
        - 1
    )

    df["lead_delta_days"] = (
        df.lead_time_days
        - df.groupby("deal_id").lead_time_days.transform("min")
    )

    df["quantity_log"] = np.log1p(df.quantity)

    return df


# ----------------------------------------------------------------------
# 3. Build sklearn ColumnTransformer
# ----------------------------------------------------------------------
def build_preprocess() -> ColumnTransformer:
    num_cols = [
        "unit_price",
        "lead_time_days",
        "quoted_margin_pct",
        "price_delta_pct",
        "lead_delta_days",
        "quantity_log",
        "on_time_rate",
    ]
    cat_cols = [
        "product_type",
        "tier",
        "region",
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


# ----------------------------------------------------------------------
# 4. Convenience function → X, y
# ----------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = add_derived_columns(df)

    # label: accepted (1) else 0
    y = df["accepted"].fillna(0).astype(int)

    # features: drop labels & leak columns
    X = df.drop(
        columns=[
            "accepted",
            "accepted_supplier_id",
            "offer_id",
            "score",
            "close_ts",
            "submitted_ts",
            "offer_ts",
        ],
        errors="ignore",
    )
    return X, y