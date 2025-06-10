# src/synthetic_generator.py
"""
Synthetic data generator for the HelloPrint quotation MVP.

Creates four CSVs under data/synthetic/:
    suppliers.csv
    deals.csv
    supplier_offers.csv
    deal_outcome.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from faker import Faker


class SyntheticGenerator:
    """Generate a realistic deal / RFQ history in four tables."""

    PRODUCTS = ("flyer", "poster", "t-shirt")

    def __init__(
        self,
        root: str | Path = "data/synthetic",
        n_suppliers: int = 25,
        n_deals: int = 2_000,
        rng_seed: int = 42,
    ) -> None:
        self.out_dir = Path(root)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.n_suppliers = n_suppliers
        self.n_deals = n_deals
        self.fake = Faker()
        Faker.seed(rng_seed)

        self.rng = np.random.default_rng(rng_seed)

        # dataframes populated later
        self.suppliers: pd.DataFrame
        self.deals: pd.DataFrame
        self.offers: pd.DataFrame
        self.outcome: pd.DataFrame

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        """Generate all tables and write them to CSV."""
        self.suppliers = self._build_suppliers()
        self.deals = self._build_deals()
        self.offers = self._build_offers()
        self.outcome = self._build_outcomes()

        self._save_csv(self.suppliers, "suppliers.csv")
        self._save_csv(self.deals, "deals.csv")
        self._save_csv(self.offers, "supplier_offers.csv")
        self._save_csv(self.outcome, "deal_outcome.csv")

        print("✓ Synthetic dataset generated in", self.out_dir)

    # --------------------------------------------------------------------- #
    # private helpers
    # --------------------------------------------------------------------- #
    def _build_suppliers(self) -> pd.DataFrame:
        rows: list[dict] = []
        for sid in range(self.n_suppliers):
            rows.append(
                {
                    "supplier_id": sid,
                    "region": self.fake.country_code(),
                    "tier": self.rng.choice(list("ABC"), p=[0.5, 0.3, 0.2]),
                    "on_time_rate": round(self.rng.uniform(0.85, 0.99), 2),
                    "product_capability": self.rng.choice(self.PRODUCTS),
                }
            )
        return pd.DataFrame(rows)

    def _build_deals(self) -> pd.DataFrame:
        rows: list[dict] = []
        start = pd.Timestamp("2024-01-01")
        for _ in range(self.n_deals):
            rows.append(
                {
                    "deal_id": self.fake.uuid4(),
                    "customer_id": self.fake.uuid4(),
                    "product_type": self.rng.choice(self.PRODUCTS),
                    "quantity": int(round(self.rng.lognormal(6, 0.4))),
                    "submitted_ts": start
                    + pd.Timedelta(hours=int(self.rng.integers(0, self.n_deals))),
                }
            )
        return pd.DataFrame(rows)

    def _build_offers(self) -> pd.DataFrame:
        """Generate supplier_offers rows (3–6 offers per deal, if available)."""
        rows: List[dict] = []

        for _, deal in self.deals.iterrows():
            # filter suppliers that can produce this product
            eligible = self.suppliers[
                self.suppliers.product_capability == deal.product_type
            ]

            if eligible.empty:
                # no supplier makes this product → skip deal
                continue

            # decide how many offers (3–6), but cap by availability
            k_requested = int(self.rng.integers(3, 7))
            k = min(k_requested, len(eligible))

            sample = self.rng.choice(
                eligible.supplier_id, size=k, replace=False
            )

            base_price = self.rng.normal(1.5, 0.2)

            # create one offer row per sampled supplier
            for sid in sample:
                rows.append(
                    {
                        "offer_id": self.fake.uuid4(),
                        "deal_id": deal.deal_id,
                        "supplier_id": int(sid),
                        "unit_price": round(base_price * self.rng.uniform(0.9, 1.15), 3),
                        "lead_time_days": int(self.rng.integers(3, 14)),
                        "quoted_margin_pct": round(self.rng.uniform(0.18, 0.30), 3),
                        "offer_ts": pd.Timestamp(deal.submitted_ts)
                        + pd.Timedelta(hours=int(self.rng.integers(2, 48))),
                    }
                )

        return pd.DataFrame(rows)

    def _build_outcomes(self) -> pd.DataFrame:
        out_rows: list[dict] = []
        for did, grp in self.offers.groupby("deal_id"):
            grp = grp.copy()
            grp["score"] = grp.unit_price.rank() + grp.lead_time_days.rank()
            winner = grp.loc[grp.score.idxmin()]
            accepted_flag = int(self.rng.random() < 0.6)  # 60 % accept

            out_rows.append(
                {
                    "deal_id": did,
                    "accepted": accepted_flag,
                    "accepted_supplier_id": winner.supplier_id if accepted_flag else None,
                    "close_ts": winner.offer_ts + pd.Timedelta(hours=5),
                }
            )
        return pd.DataFrame(out_rows)

    # save utility
    def _save_csv(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.out_dir / name, index=False)


# --------------------------------------------------------------------------
# CLI entry-point for quick manual generationº
# --------------------------------------------------------------------------
if __name__ == "__main__":
    SyntheticGenerator().run()