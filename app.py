import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List

st.set_page_config(page_title="6‑Pillar Investment Evaluator", layout="wide")

# --- Helpers ---
PILLARS = [
    ("Return Potential", "Expected return vs benchmark, consistency of returns, growth drivers"),
    ("Risk Profile", "Volatility, downside scenarios, concentration & leverage"),
    ("Liquidity", "Redemption terms, lockups, secondary market availability"),
    ("Tax Efficiency", "Tax treatment, withholding, tax-loss harvesting opportunities"),
    ("Manager & Governance", "Track record, alignment of interest, reporting & audits"),
    ("Alignment to Objectives", "Fit with investor goals, constraints, horizon"),
]

RECOMM_BANDS = [
    (25, 30, "Strong Buy / Anchor allocation"),
    (18, 24, "Opportunistic / Limited allocation"),
    (0, 17, "Pass"),
]

RED_FLAGS = [
    "Unrealistic IRR / return claims",
    "Very high fees (management/carry)",
    "No audited NAVs / poor reporting",
    "Excessive structural complexity",
    "Over-concentration in single asset or sector",
]


def compute_total_score(scores: List[int]) -> int:
    return int(sum(scores))


def band_from_score(total: int) -> str:
    for lo, hi, name in RECOMM_BANDS:
        if lo <= total <= hi:
            return name
    # fallback
    return "Pass"


def format_pct(x: float) -> str:
    return f"{x:.1f}%"


# --- UI ---
st.title("6‑Pillar Investment Evaluator")
st.markdown("Use this interactive tool to score an investment across six pillars, note red flags, build allocation suggestions and run simple stress tests.")

# Layout columns for pillars and quick results
with st.container():
    cols = st.columns((3, 2))

    with cols[0]:
        st.subheader("1) Core Pillars Scoring")
        st.write("Slide each pillar from 1 (weak) to 5 (excellent). Captions help you remember the key metrics to consider.")

        pillar_scores = []
        for i, (name, caption) in enumerate(PILLARS):
            st.markdown(f"**{name}**")
            c = st.slider(f"Score: {name}", 1, 5, 3, key=f"pillar_{i}")
            st.caption(caption)
            pillar_scores.append(c)

    with cols[1]:
        st.subheader("Summary & Recommendation")
        total = compute_total_score(pillar_scores)
        st.metric("Total Pillar Score (6–30)", f"{total}")

        # Recommendation by band
        rec = band_from_score(total)

        # Red flags area will override rec if any selected (we will compute later)
        st.write("**Recommendation band (before red-flag override):**")
        st.info(rec)

        # Show a short textual guide of bands
        st.write("**Bands:**")
        st.markdown(
            "- 25–30: Strong Buy / Anchor allocation  
- 18–24: Opportunistic / Limited allocation  
- <18: Pass"
        )

# 2) Red Flags
st.markdown("---")
st.subheader("2) Red Flags")
st.write("Select any red flags you find relevant. If any red flag is selected the tool will show a warning and the overall recommendation is overridden to **Pass**.")

selected_flags = []
cols_flags = st.columns(2)
with cols_flags[0]:
    for i, flag in enumerate(RED_FLAGS[:3]):
        if st.checkbox(flag, key=f"flag_{i}"):
            selected_flags.append(flag)
with cols_flags[1]:
    for i, flag in enumerate(RED_FLAGS[3:], start=3):
        if st.checkbox(flag, key=f"flag_{i}"):
            selected_flags.append(flag)

if selected_flags:
    st.warning("One or more red flags selected — recommendation will be overridden to **Pass**. Review flags before proceeding.")
    st.write("**Selected flags:**")
    for f in selected_flags:
        st.write(f"- {f}")
    final_recommendation = "Pass (red flag)"
else:
    final_recommendation = band_from_score(total)

st.success(f"Final recommendation: **{final_recommendation}**")

# 3) Portfolio Construction
st.markdown("---")
st.subheader("3) Portfolio Construction")
st.write("Create an allocation across the four portfolio buckets. Raw sliders accept 0–100; allocations are normalized automatically to sum to 100%.")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    raw_core = st.slider("Core (raw)", 0, 100, 40, key="raw_core")
with pc2:
    raw_sat = st.slider("Satellite (raw)", 0, 100, 30, key="raw_sat")
with pc3:
    raw_alt = st.slider("Alternatives (raw)", 0, 100, 20, key="raw_alt")
with pc4:
    raw_tac = st.slider("Tactical (raw)", 0, 100, 10, key="raw_tac")

raws = np.array([raw_core, raw_sat, raw_alt, raw_tac], dtype=float)
if raws.sum() == 0:
    # default equal weights if user sets all to zero
    weights = np.array([0.25, 0.25, 0.25, 0.25])
else:
    weights = raws / raws.sum()

labels = ["Core", "Satellite", "Alternatives", "Tactical"]
weights_pct = [w * 100 for w in weights]

st.write("**Normalized allocation (sums to 100%)**")
colsw = st.columns(4)
for i, col in enumerate(colsw):
    col.metric(labels[i], format_pct(weights_pct[i]))

# Pie chart of portfolio weights
fig = go.Figure(data=[go.Pie(labels=labels, values=weights_pct, hole=0.4)])
fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
st.plotly_chart(fig, use_container_width=True)

# 4) Stress Testing / Monte Carlo Simulation
st.markdown("---")
st.subheader("4) Stress Testing & Monte Carlo Simulation")
st.write("A simple Monte Carlo simulator (geometric‑like yearly returns). All results are approximate and for illustrative use only.")

mc_col1, mc_col2 = st.columns(2)
with mc_col1:
    expected_return = st.number_input("Expected annual return (as %)", value=8.0, step=0.1)
    volatility = st.number_input("Annual volatility / stdev (as %)", value=12.0, step=0.1)
    years = st.number_input("Years", min_value=1, max_value=50, value=10)
with mc_col2:
    n_paths = st.number_input("Number of simulated paths", min_value=100, max_value=20000, value=1000)
    start_value = st.number_input("Starting portfolio value", value=1000.0, step=100.0)

run_mc = st.button("Run Simulation")

if run_mc:
    # Convert to decimal
    mu = expected_return / 100.0
    sigma = volatility / 100.0
    T = int(years)
    n = int(n_paths)

    # simulate yearly returns using normal approximation
    # shape: (n_paths, years)
    rng = np.random.default_rng()
    rets = rng.normal(loc=mu, scale=sigma, size=(n, T))
    # build growth paths: cumulative product of (1 + ret)
    paths = start_value * np.cumprod(1 + rets, axis=1)

    years_idx = np.arange(1, T + 1)

    # compute median path
    median_path = np.median(paths, axis=0)

    # compute approximate max drawdown for median path
    running_max = np.maximum.accumulate(median_path)
    drawdowns = (running_max - median_path) / running_max
    max_drawdown = np.max(drawdowns)

    # also compute worst max drawdown across all paths (approx)
    running_max_all = np.maximum.accumulate(paths, axis=1)
    drawdowns_all = (running_max_all - paths) / running_max_all
    worst_path_mdd = np.max(drawdowns_all)

    st.write(f"**Median terminal value:** {median_path[-1]:,.2f}")
    st.write(f"**Approx median max drawdown:** {max_drawdown:.2%}")
    st.write(f"**Worst path max drawdown (across simulated paths):** {worst_path_mdd:.2%}")

    # Plot: overlay small-sample of paths + median
    sample_n = min(200, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)

    fig_mc = go.Figure()
    for idx in sample_idx:
        fig_mc.add_trace(go.Scatter(x=years_idx, y=paths[idx, :], mode="lines", opacity=0.08, showlegend=False))
    fig_mc.add_trace(go.Scatter(x=years_idx, y=median_path, mode="lines", name="Median path", line=dict(width=3)))
    fig_mc.update_layout(title="Monte Carlo growth paths (sample) with median", xaxis_title="Year", yaxis_title="Portfolio value", height=450, margin=dict(t=40))
    st.plotly_chart(fig_mc, use_container_width=True)

    # show percentile table for terminal values
    term_vals = paths[:, -1]
    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(term_vals, pct)
    df_pct = pd.DataFrame({"Percentile": pct, "Terminal value": pct_vals})
    st.write("**Terminal value percentiles**")
    st.dataframe(df_pct)

# Footer / additional controls
st.markdown("---")
with st.expander("Notes & assumptions"):
    st.write(
        "This tool is a lightweight, client-side evaluator meant for quick screening and educational use.  \n"
        "• Pillar scoring is intentionally simple — map your qualitative judgement to scores 1–5.  \n"
        "• Red flags are a conservative override — if any apply, treat the opportunity with extreme caution.  \n"
        "• Monte Carlo uses a simple yearly-normal return model; real-world return distributions may be non-normal and path-dependent."
    )

st.caption("Built with Streamlit — all calculations run locally in the browser/session. No external APIs.")

