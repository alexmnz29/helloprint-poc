# src/app.py
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from inference import InferenceEngine

# ------------------------------------------------------------------ #
# Paths & colours
# ------------------------------------------------------------------ #
MODEL_PATH = Path("model.joblib")
LOGO_PATH  = Path("assets/helloprint_logo.png")

BLACK       = "#111111"
DARK_ORANGE = "#CC7A00"

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(page_title="HelloPrint • Quote Optimiser",
                   page_icon="🖨️", layout="wide")

# ------------------------------------------------------------------ #
# Minimal CSS (slider styling only; other colours via theme)
# ------------------------------------------------------------------ #
st.markdown(
    f"""
    <style>
      .stSlider .rc-slider-rail   {{ background:{DARK_ORANGE}!important; height:6px; }}
      .stSlider .rc-slider-track {{ background:{DARK_ORANGE}; height:6px; }}
      .stSlider .rc-slider-handle {{
        background:{DARK_ORANGE}; border:2px solid {DARK_ORANGE};
        width:16px; height:16px; margin-top:-5px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Header
# ------------------------------------------------------------------ #
left, mid, logo = st.columns([1, 4, 1])
with mid:
    st.markdown("<h1 style='text-align:center;margin-bottom:0.2rem;'>"
                "Quote Optimiser Demo</h1>", unsafe_allow_html=True)
with logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)
st.divider()

# ------------------------------------------------------------------ #
# Sidebar – margin floor slider
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("**Margin floor (%)**")
    margin_floor = st.slider("", 5, 40, 20) / 100

# ------------------------------------------------------------------ #
# Explanation block (always visible)
# ------------------------------------------------------------------ #
st.subheader("How this demo works")
st.markdown(
    """
**End-to-end flow (mocked)**  

1. **Upload RFQ PDF** – Treat it as the customer’s request.
2. **Mock OCR & supplier shortlist** – The demo pretends to extract product, quantity, deadline and finds compatible suppliers.  
3. **Win-probability model** – A pre-trained XGBoost model scores each offer.  
4. **Margin-constrained optimiser** – Picks the offer that maximises *p*<sub>win</sub> × margin while meeting the slider threshold.  
5. **Results** – Shows RFQ summary, ranked supplier table, selected offer and CSV download.

> All computation is local.
"""
)

# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
PRODUCTS = ["flyer", "poster", "t-shirt"]
REGIONS = ["NL", "DE", "FR", "ES"]

def mock_parse_pdf() -> dict:
    return {
        "client_id": f"C-{random.randint(100, 999)}",
        "region": random.choice(REGIONS),
        "product_type": random.choice(PRODUCTS),
        "quantity": random.choice([500, 1000, 1500, 2000]),
        "deadline": (date.today() + timedelta(days=14)).isoformat(),
    }

def mock_offers(rfq: dict, n: int = 6) -> pd.DataFrame:
    base_price = random.uniform(1.2, 1.8)
    rows = []
    for _ in range(n):
        rows.append({
            "supplier_id": random.randint(1, 30),
            "product_type": rfq["product_type"],
            "unit_price": round(base_price * random.uniform(0.9, 1.15), 2),
            "lead_time_days": random.randint(4, 10),
            "quoted_margin_pct": round(random.uniform(0.18, 0.30), 2),
            "quantity": rfq["quantity"],
            "tier": random.choice(["A", "B", "C"]),
            "region": rfq["region"],
            "on_time_rate": round(random.uniform(0.88, 0.98), 2),
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ #
# Step 1 – Upload RFQ PDF
# ------------------------------------------------------------------ #
st.subheader("1. Upload RFQ (PDF)")
pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])

if pdf_file:
    rfq = mock_parse_pdf()
    st.table(pd.DataFrame([rfq]))

# ------------------------------------------------------------------ #
# Step 2 – Request quotation
# ------------------------------------------------------------------ #
st.subheader("2. Request quotation")
if pdf_file and st.button("Calculate best offer"):
    offers_df = mock_offers(rfq)

    # ---- load model
    if not MODEL_PATH.exists():
        st.error("model.joblib not found. Run `make train` first.")
        st.stop()
    engine = InferenceEngine(MODEL_PATH, margin_floor=margin_floor)

    try:
        best, ranked = engine.select_best_offer(offers_df)
    except RuntimeError as e:
        st.error(f"🚫 {e}")
        st.stop()

    # ---- prepare ranked display
    ranked["margin %"] = (ranked["quoted_margin_pct"] * 100).round(1)
    ranked["p(win) %"] = (ranked["p_win"] * 100).round(1)
    ranked = ranked.rename(columns={
        "supplier_id": "Supplier",
        "unit_price": "€ / unit",
        "lead_time_days": "Lead-time (days)",
        "utility": "Utility",
    })
    cols = ["Supplier", "€ / unit", "Lead-time (days)", "margin %", "p(win) %", "Utility"]

    table_col, metric_col = st.columns([3, 1], gap="large")
    with table_col:
        st.subheader("Ranked offers")
        st.dataframe(ranked[cols], hide_index=True, use_container_width=True)

    with metric_col:
        st.subheader("Selected offer")

        # metrics in one horizontal row
        c_sup, c_win, c_margin = st.columns(3)
        c_sup.metric("Supplier", best["supplier_id"])
        c_win.metric("Win probability", f"{best['p_win']:.0%}")
        c_margin.metric("Margin", f"{best['margin_pct']:.0%}")

        # rationale expander
        with st.expander("Why this offer?", expanded=False):
            st.markdown(
                f"""
* **Highest expected value** – Maximises *p*<sub>win</sub> ({best['p_win']:.0%}) × margin ({best['margin_pct']:.0%}).  
* **Meets margin floor** – Above the minimum of **{int(margin_floor*100)} %**.  
* Alternatives have lower expected value or fail the margin rule.
""",
                unsafe_allow_html=True,
            )

        # download button
        csv = ranked[cols].to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "ranked_offers.csv",
                           mime="text/csv", use_container_width=True)
else:
    st.info("Upload a PDF and click *Calculate best offer* to see results.")