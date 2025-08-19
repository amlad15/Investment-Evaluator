# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Mutual Fund Evaluator (India)", layout="wide")

API_SEARCH = "https://api.mfapi.in/mf/search"
API_SCHEME = "https://api.mfapi.in/mf/{}"

# --- Helpers ---
def mf_search(query: str) -> List[Dict]:
    """Search MFAPI for a query term. Returns list of {schemeCode, schemeName}."""
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def fetch_scheme(scheme_code: str) -> Dict:
    """Fetch scheme data JSON (includes NAV history under 'data')."""
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch scheme {scheme_code}: {e}")
        return {}

def nav_history_to_df(nav_json: List[Dict]) -> pd.DataFrame:
    """Convert MFAPI 'data' to DataFrame with date and nav (float)."""
    df = pd.DataFrame(nav_json)
    # expected keys: 'date' and 'nav' (sometimes 'nav' may be like '12.345' strings)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # parse date (MFAPI uses dd-mm-yyyy)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    return df

def compute_cagr(df: pd.DataFrame) -> float:
    if df.shape[0] < 2:
        return float("nan")
    start = df["nav"].iloc[0]
    end = df["nav"].iloc[-1]
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    if days <= 0 or start <= 0:
        return float("nan")
    years = days / 365.25
    cagr = (end / start) ** (1 / years) - 1
    return cagr

def compute_annual_vol(df: pd.DataFrame) -> float:
    # daily returns from NAV; annualize by sqrt(252)
    if df.shape[0] < 2:
        return float("nan")
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    vol = df["ret"].std(ddof=0) * np.sqrt(252)
    return vol

def compute_sharpe(cagr: float, vol: float, rf: float = 0.03) -> float:
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return (cagr - rf) / vol

def auto_score_from_metrics(cagr: float, vol: float, df: pd.DataFrame) -> List[int]:
    """
    Produce an automatic 1-5 score for each pillar:
    1. Return Potential (based on cagr)
    2. Risk Profile (based on vol)
    3. Liquidity (heuristic from NAV history length)
    4. Tax Efficiency (generic default 3)
    5. Manager & Governance (default 3)
    6. Alignment to Objectives (default 3)
    """
    scores = []
    # Return Potential
    if not np.isfinite(cagr):
        scores.append(3)
    elif cagr >= 0.20:
        scores.append(5)
    elif cagr >= 0.12:
        scores.append(4)
    elif cagr >= 0.06:
        scores.append(3)
    elif cagr >= 0.0:
        scores.append(2)
    else:
        scores.append(1)

    # Risk Profile: lower vol => higher score
    if not np.isfinite(vol):
        scores.append(3)
    elif vol <= 0.10:
        scores.append(5)
    elif vol <= 0.20:
        scores.append(4)
    elif vol <= 0.35:
        scores.append(3)
    elif vol <= 0.50:
        scores.append(2)
    else:
        scores.append(1)

    # Liquidity: if NAV history long and many points -> better liquidity heuristic
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days if df.shape[0] >= 2 else 0
    if days >= 3650:
        scores.append(5)
    elif days >= 1825:
        scores.append(4)
    elif days >= 365:
        scores.append(3)
    elif days >= 180:
        scores.append(2)
    else:
        scores.append(1)

    # Tax Efficiency: default neutral (user should edit)
    scores.append(3)
    # Manager & Governance: default neutral
    scores.append(3)
    # Alignment to Objectives: default neutral
    scores.append(3)

    return scores

def band_from_score(total: int) -> str:
    if total >= 25:
        return "Strong Buy / Anchor allocation"
    if total >= 18:
        return "Opportunistic / Limited allocation"
    return "Pass"

def monte_carlo_sim(mu: float, sigma: float, years: int, n_paths: int, start_value: float):
    """Simulate geometric returns using annual mu & sigma; returns array (n_paths, years+1)."""
    dt = 1.0  # yearly steps (we use yearly steps for simplicity)
    rng = np.random.default_rng()
    steps = years
    # Use normal approximation to annual returns: r_t ~ N(mu, sigma)
    rets = rng.normal(loc=mu, scale=sigma, size=(n_paths, steps))
    paths = np.empty((n_paths, steps + 1), dtype=float)
    paths[:, 0] = start_value
    for t in range(steps):
        paths[:, t + 1] = paths[:, t] * (1 + rets[:, t])
    return paths

# --- UI ---
st.title("Mutual Fund Evaluator (India) — Search, Auto-score & Monte Carlo")
st.markdown(
    "Search for an Indian mutual fund (MFAPI) by name, fetch NAV history, get automatic metrics & suggested 6-pillar scores — editable — and run stress tests."
)

# Sidebar: Monte Carlo defaults and global settings
with st.sidebar:
    st.header("Simulation & Settings")
    rf_rate = st.number_input("Risk-free rate (annual, %)", value=3.0, step=0.1) / 100.0
    default_years = st.number_input("Default MC years", min_value=1, max_value=30, value=10)
    default_paths = st.number_input("Default number of MC paths", min_value=100, max_value=5000, value=1000, step=100)

# 1) SEARCH
st.subheader("1) Search fund by name")
query = st.text_input("Enter fund name (e.g., 'Bandhan Small Cap')", value="")
search_button = st.button("Search")

search_results = []
if search_button and query.strip():
    with st.spinner("Searching MFAPI..."):
        search_results = mf_search(query.strip())
    if not search_results:
        st.warning("No results found. Try a broader query or check spelling.")
else:
    # keep empty until search pressed
    search_results = []

if search_results:
    # show dropdown of results with user-friendly labels
    options = [f"{r.get('schemeName')} — {r.get('schemeCode')}" for r in search_results]
    choice = st.selectbox("Select a scheme from results", options)
    # parse scheme code
    selected_idx = options.index(choice)
    selected_scheme = search_results[selected_idx]
    scheme_code = str(selected_scheme.get("schemeCode"))
    scheme_name = selected_scheme.get("schemeName")
    st.success(f"Selected: {scheme_name} (code: {scheme_code})")

    # Fetch scheme data
    with st.spinner("Fetching NAV history..."):
        scheme_json = fetch_scheme(scheme_code)

    if scheme_json and "data" in scheme_json:
        df_nav = nav_history_to_df(scheme_json["data"])
        if df_nav.empty:
            st.error("NAV history empty after parsing. Cannot analyze.")
        else:
            # show basic info
            st.subheader("Fund overview & NAV")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_nav["date"], y=df_nav["nav"], mode="lines", name="NAV"))
                fig.update_layout(title="NAV history", xaxis_title="Date", yaxis_title="NAV", height=350)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.write(f"**Scheme:** {scheme_name}")
                st.write(f"**Scheme code:** {scheme_code}")
                # MFAPI sometimes includes meta fields; show if present
                meta = scheme_json.get("meta", {})
                if meta:
                    # show common meta keys
                    for k in ["plan", "schemeType", "fund_house", "schemeCategory", "schemeCode"]:
                        if k in meta:
                            st.write(f"**{k}:** {meta.get(k)}")
                st.write(f"Data points: {df_nav.shape[0]}")
                st.write(f"From {df_nav['date'].iloc[0].date()} to {df_nav['date'].iloc[-1].date()}")

            # compute metrics
            cagr = compute_cagr(df_nav)
            vol = compute_annual_vol(df_nav)
            sharpe = compute_sharpe(cagr, vol, rf=rf_rate)

            st.subheader("Calculated metrics")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("CAGR (annual)", f"{cagr:.2%}" if np.isfinite(cagr) else "N/A")
            mcol2.metric("Annualized volatility", f"{vol:.2%}" if np.isfinite(vol) else "N/A")
            mcol3.metric("Sharpe (rf {0:.1%})".format(rf_rate), f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")

            # Auto-score the 6 pillars and allow user override
            st.markdown("---")
            st.subheader("6-Pillar Scoring (auto-suggested, editable)")
            auto_scores = auto_score_from_metrics(cagr, vol, df_nav)
            PILLARS = [
                ("Return Potential", "Expected return vs benchmark, consistency of returns, growth drivers"),
                ("Risk Profile", "Volatility, downside scenarios, concentration & leverage"),
                ("Liquidity", "Redemption terms, lockups, secondary market availability"),
                ("Tax Efficiency", "Tax treatment and structure (manual review recommended)"),
                ("Manager & Governance", "Fund house track record, disclosures, audit quality (manual)"),
                ("Alignment to Objectives", "Fit with investor goals, horizon, constraints (manual)"),
            ]

            cols = st.columns(3)
            pillar_scores = []
            for i, (name, caption) in enumerate(PILLARS):
                # place two sliders per column to keep compact
                col = cols[i % 3]
                # default to auto score
                s = col.slider(f"{name}", min_value=1, max_value=5, value=int(auto_scores[i]), key=f"pillar_{i}")
                col.caption(caption)
                pillar_scores.append(s)

            total_score = sum(pillar_scores)
            band = band_from_score(total_score)
            st.write("")
            st.markdown(f"**Total Pillar Score:** {total_score} / 30")
            st.info(f"Recommendation (before red-flag override): {band}")

            # Auto-detected red flags (heuristics)
            st.subheader("Red Flags (auto-detected + manual)")
            auto_flags = []
            # 1) Limited track record
            days = (df_nav["date"].iloc[-1] - df_nav["date"].iloc[0]).days
            if days < 365:
                auto_flags.append("Limited track record (<1 year)")
            # 2) Very high volatility
            if np.isfinite(vol) and vol > 0.50:
                auto_flags.append("Very high volatility ( > 50% pa )")
            # 3) Unrealistic CAGR
            if np.isfinite(cagr) and cagr > 1.0:
                auto_flags.append("Unrealistic return claims (CAGR > 100% pa)")
            # 4) Low data points
            if df_nav.shape[0] < 60:
                auto_flags.append("Few NAV observations (may be thinly traded / new fund)")

            # present auto flags first
            selected_flags = []
            if auto_flags:
                st.warning("Auto-detected flags (please review)")
                for f in auto_flags:
                    st.write(f"- {f}")
                    # also pre-check these in the manual checklist
                    selected_flags.append(f)

            # Manual flags (user can also check)
            manual_flags_list = [
                "Unrealistic IRR / return claims",
                "Very high fees (management/carry)",
                "No audited NAVs / poor reporting",
                "Excessive structural complexity",
                "Over-concentration in single asset or sector",
                "Limited track record / short history",
                "High exit loads / redemption restrictions",
            ]
            st.write("Select manual flags (if any):")
            cols_flags = st.columns(2)
            for i, flag in enumerate(manual_flags_list):
                if cols_flags[i % 2].checkbox(flag, key=f"manual_flag_{i}"):
                    selected_flags.append(flag)

            # Combine auto + manual flags, remove duplicates
            selected_flags = list(dict.fromkeys(selected_flags))

            if selected_flags:
                st.write("**Active flags:**")
                for f in selected_flags:
                    st.write(f"- {f}")

            # Final recommendation with override
            if selected_flags:
                final_rec = "Pass (red flag)"
            else:
                final_rec = band

            st.success(f"Final recommendation: **{final_rec}**")

            # Portfolio Construction quick section
            st.markdown("---")
            st.subheader("Quick Portfolio Construction")
            r1, r2, r3, r4 = st.columns(4)
            raw_core = r1.slider("Core (raw)", 0, 100, 40, key="raw_core")
            raw_sat = r2.slider("Satellite (raw)", 0, 100, 30, key="raw_sat")
            raw_alt = r3.slider("Alternatives (raw)", 0, 100, 20, key="raw_alt")
            raw_tac = r4.slider("Tactical (raw)", 0, 100, 10, key="raw_tac")
            raws = np.array([raw_core, raw_sat, raw_alt, raw_tac], dtype=float)
            if raws.sum() == 0:
                weights = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                weights = raws / raws.sum()
            labels = ["Core", "Satellite", "Alternatives", "Tactical"]
            weights_pct = weights * 100
            sw = st.columns(4)
            for i, ccol in enumerate(sw):
                ccol.metric(labels[i], f"{weights_pct[i]:.1f}%")
            # pie
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=weights_pct, hole=0.35)])
            fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)

            # Monte Carlo simulation controls
            st.markdown("---")
            st.subheader("Stress Testing / Monte Carlo Simulation")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_years = st.number_input("Years to simulate", min_value=1, max_value=30, value=default_years)
                sim_paths = st.number_input("Number of simulated paths", min_value=100, max_value=5000, value=default_paths)
                start_value = st.number_input("Starting portfolio value (for sim)", value=10000.0, step=100.0)
            with sim_col2:
                # let user override mu & sigma or use computed
                use_computed = st.checkbox("Use computed CAGR/Vol for sim", value=True)
                if not use_computed:
                    mu_input = st.number_input("Sim expected annual return (as %)", value=8.0, step=0.1) / 100.0
                    sigma_input = st.number_input("Sim annual volatility (as %)", value=12.0, step=0.1) / 100.0
                else:
                    mu_input = cagr if np.isfinite(cagr) else 0.08
                    sigma_input = vol if np.isfinite(vol) else 0.12

            run_sim = st.button("Run Monte Carlo")

            if run_sim:
                n = int(sim_paths)
                y = int(sim_years)
                mu = float(mu_input)
                sigma = float(sigma_input)
                if not np.isfinite(mu):
                    st.error("Invalid expected return for simulation.")
                elif not np.isfinite(sigma):
                    st.error("Invalid volatility for simulation.")
                else:
                    with st.spinner("Running Monte Carlo..."):
                        paths = monte_carlo_sim(mu=mu, sigma=sigma, years=y, n_paths=n, start_value=start_value)
                        # median path
                        median_path = np.median(paths, axis=0)
                        # compute drawdowns on median path
                        running_max = np.maximum.accumulate(median_path)
                        drawdowns = (running_max - median_path) / running_max
                        mdd = np.max(drawdowns)

                        # worst path mdd
                        running_max_all = np.maximum.accumulate(paths, axis=1)
                        drawdowns_all = (running_max_all - paths) / running_max_all
                        worst_mdd = np.nanmax(drawdowns_all)

                        st.write(f"**Median terminal value:** {median_path[-1]:,.2f}")
                        st.write(f"**Approx median max drawdown:** {mdd:.2%}")
                        st.write(f"**Worst path max drawdown (simulated):** {worst_mdd:.2%}")

                        # plot sample of paths and median
                        sample_n = min(200, n)
                        rng = np.random.default_rng()
                        idx = rng.choice(n, size=sample_n, replace=False)
                        fig_mc = go.Figure()
                        x = list(range(0, y + 1))
                        for i_s in idx:
                            fig_mc.add_trace(go.Scatter(x=x, y=paths[i_s, :], mode="lines", line=dict(width=1), opacity=0.08, showlegend=False))
                        fig_mc.add_trace(go.Scatter(x=x, y=median_path, mode="lines", name="Median", line=dict(width=3)))
                        fig_mc.update_layout(title="Monte Carlo simulated growth (sample paths + median)", xaxis_title="Year", yaxis_title="Value", height=450)
                        st.plotly_chart(fig_mc, use_container_width=True)

                        # percentiles of terminal values
                        term_vals = paths[:, -1]
                        pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                        pct_vals = np.percentile(term_vals, pct)
                        df_pct = pd.DataFrame({"Percentile": pct, "Terminal value": pct_vals})
                        st.write("Terminal value percentiles")
                        st.dataframe(df_pct)

            # end scheme processing
    else:
        st.error("Unexpected response from MFAPI or missing 'data'.")
else:
    st.info("Enter a fund name and press Search to begin.")
