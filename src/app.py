# src/app.py — slider in sidebar, uploader in main panel
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from inference import InferenceEngine

# ------------------------------------------------------------------ #
# Paths & colours
# ------------------------------------------------------------------ #
MODEL_PATH = Path("model.joblib")
LOGO_PATH  = Path("assets/helloprint_logo.png")

BLACK        = "#111111"
DARK_ORANGE  = "#CC7A00"  # for slider & button

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(page_title="HelloPrint • Quote Optimiser",
                   page_icon="🖨️", layout="wide")

# ------------------------------------------------------------------ #
# Global + component CSS
# ------------------------------------------------------------------ #
st.markdown(
    f"""
    <style>
      .stApp {{ background:#FFFFFF; color:{BLACK}; font-family:Inter,sans-serif; }}

      /* sidebar backdrop + text */
      section[data-testid="stSidebar"] > div:first-child {{ background:#2B2B2B !important; }}
      section[data-testid="stSidebar"] > div:first-child * {{ color:#F5F5F5 !important; }}

      /* slider (orange rail/track/handle) */
      .stSlider .rc-slider-rail   {{ background:{DARK_ORANGE} !important; height:6px; }}
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
c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    st.markdown("<h1 style='text-align:center;margin-bottom:0.2rem;'>"
                "Quote&nbsp;Optimiser&nbsp;Demo</h1>", unsafe_allow_html=True)
with c3:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)
st.divider()

# ------------------------------------------------------------------ #
# SIDEBAR – only the slider
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("**Margin floor (%)**")
    margin_floor = st.slider("", 5, 40, 15) / 100

# ------------------------------------------------------------------ #
# MAIN  –  explanation + uploader side-by-side
# ------------------------------------------------------------------ #
explain_col, upload_col = st.columns([3, 1], gap="large")

with explain_col:
    st.subheader("How this demo works")
    st.markdown(
        """
1. **Upload data** – Provide a JSON file with one or more supplier offers for the *same* RFQ.  
2. **Feature engineering** – The app derives price competitiveness, lead-time delta, quantity log scale and supplier reliability.  
3. **Probability model** – A trained XGBoost model estimates the acceptance probability (*p*<sub>win</sub>) for each offer.  
4. **Optimiser** – A linear solver picks a single offer that maximises *p*<sub>win</sub> × margin while respecting the margin slider.  
5. **Results** – The ranked table shows all offers; the panel on the right highlights the chosen supplier and metrics.

*Everything runs locally; no external services are called.*
        """
    )

with upload_col:
    st.subheader("Upload offers")
    uploaded = st.file_uploader(
        label="Browse&nbsp;files",
        label_visibility="hidden",
        type=["json"],
        accept_multiple_files=False,
        help="Upload a JSON file exported from your RFQ parser"
    )

# ------------------------------------------------------------------ #
# Load model
# ------------------------------------------------------------------ #
if not MODEL_PATH.exists():
    st.error("model.joblib not found.  Run `make train` first.")
    st.stop()

engine = InferenceEngine(MODEL_PATH, margin_floor=margin_floor)

# ------------------------------------------------------------------ #
# MAIN LOGIC
# ------------------------------------------------------------------ #
if uploaded:
    offers_df = pd.DataFrame(json.load(uploaded))

    try:
        best, ranked = engine.select_best_offer(offers_df)
    except RuntimeError as e:
        st.error(f"🚫 {e}")
        st.stop()

    ranked["margin %"] = (ranked["quoted_margin_pct"] * 100).round(1)
    ranked["p(win) %"] = (ranked["p_win"] * 100).round(1)
    ranked = ranked.rename(columns={
        "supplier_id": "Supplier",
        "unit_price":  "€ / unit",
        "lead_time_days": "Lead-time (days)",
        "utility": "Utility",
    })
    cols = ["Supplier", "€ / unit", "Lead-time (days)", "margin %", "p(win) %", "Utility"]

    tbl_col, met_col = st.columns([3, 1], gap="large")
    with tbl_col:
        st.subheader("Ranked offers")
        st.dataframe(ranked[cols], hide_index=True, use_container_width=True)

    with met_col:
        st.subheader("Selected offer")
        st.metric("Supplier", best["supplier_id"])
        st.metric("Win probability", f"{best['p_win']:.0%}")
        st.metric("Margin", f"{best['margin_pct']:.0%}")

        csv = ranked.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "ranked_offers.csv",
                           mime="text/csv", use_container_width=True)