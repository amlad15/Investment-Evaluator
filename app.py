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
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def fetch_scheme(scheme_code: str) -> Dict:
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch scheme {scheme_code}: {e}")
        return {}

def nav_history_to_df(nav_json: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(nav_json)
    df = df.rename(columns={c: c.strip() for c in df.columns})
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
    return (end / start) ** (1 / years) - 1

def compute_annual_vol(df: pd.DataFrame) -> float:
    if df.shape[0] < 2:
        return float("nan")
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    return df["ret"].std(ddof=0) * np.sqrt(252)

def compute_sharpe(cagr: float, vol: float, rf: float = 0.03) -> float:
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return (cagr - rf) / vol

def auto_score_from_metrics(cagr: float, vol: float, df: pd.DataFrame, meta: Dict) -> List[int]:
    scores = []
    # Return Potential
    if not np.isfinite(cagr):
        scores.append(3)
    elif cagr >= 0.20: scores.append(5)
    elif cagr >= 0.12: scores.append(4)
    elif cagr >= 0.06: scores.append(3)
    elif cagr >= 0.0: scores.append(2)
    else: scores.append(1)
    # Risk Profile
    if not np.isfinite(vol):
        scores.append(3)
    elif vol <= 0.10: scores.append(5)
    elif vol <= 0.20: scores.append(4)
    elif vol <= 0.35: scores.append(3)
    elif vol <= 0.50: scores.append(2)
    else: scores.append(1)
    # Liquidity
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days if df.shape[0] >= 2 else 0
    if days >= 3650: scores.append(5)
    elif days >= 1825: scores.append(4)
    elif days >= 365: scores.append(3)
    elif days >= 180: scores.append(2)
    else: scores.append(1)
    # Tax Efficiency from schemeCategory
    cat = meta.get("schemeCategory","").lower()
    if "elss" in cat: scores.append(2)
    else: scores.append(3)
    # Manager & Governance heuristic
    top_houses = ["HDFC", "ICICI", "SBI", "Bandhan"]
    house = meta.get("fund_house","").upper()
    scores.append(5 if any(th in house for th in top_houses) else 3)
    # Alignment to Objectives default
    scores.append(3)
    return scores

def band_from_score(total: int) -> str:
    if total >= 25: return "Strong Buy / Anchor allocation"
    if total >= 18: return "Opportunistic / Limited allocation"
    return "Pass"

def monte_carlo(mu: float, sigma: float, years: int, n_paths: int, start_value: float):
    dt = 1.0
    rng = np.random.default_rng()
    steps = years
    rets = rng.normal(loc=mu, scale=sigma, size=(n_paths, steps))
    paths = np.empty((n_paths, steps+1), dtype=float)
    paths[:,0] = start_value
    for t in range(steps):
        paths[:,t+1] = paths[:,t]*(1+rets[:,t])
    return paths

# --- Session State Init ---
if "selected_scheme" not in st.session_state: st.session_state.selected_scheme = None
if "df_nav" not in st.session_state: st.session_state.df_nav = None
if "metrics" not in st.session_state: st.session_state.metrics = {}
if "pillar_scores" not in st.session_state: st.session_state.pillar_scores = []
if "red_flags" not in st.session_state: st.session_state.red_flags = []
if "portfolio" not in st.session_state: st.session_state.portfolio = [0.4,0.3,0.2,0.1]
if "mc_results" not in st.session_state: st.session_state.mc_results = None

# --- Sidebar ---
with st.sidebar:
    st.header("Simulation & Settings")
    rf_rate = st.number_input("Risk-free rate (%)", value=3.0)/100.0
    default_years = st.number_input("Default MC years", min_value=1, max_value=30, value=10)
    default_paths = st.number_input("Default MC paths", min_value=100, max_value=5000, value=1000)

# --- Main UI ---
st.title("Mutual Fund Evaluator (India)")
st.markdown("Search a fund, auto-score 6 pillars, detect red flags, portfolio, and run Monte Carlo simulation.")

query = st.text_input("Enter fund name", value="")
search_button = st.button("Search")

search_results = []
if search_button and query.strip():
    with st.spinner("Searching MFAPI..."):
        search_results = mf_search(query.strip())

if search_results:
    options = [f"{r['schemeName']} — {r['schemeCode']}" for r in search_results]
    choice = st.selectbox("Select a scheme", options, index=0)
    selected_idx = options.index(choice)
    selected_scheme = search_results[selected_idx]
    scheme_code = str(selected_scheme["schemeCode"])
    scheme_name = selected_scheme["schemeName"]

    # Load scheme if different
    if st.session_state.selected_scheme != scheme_code:
        st.session_state.selected_scheme = scheme_code
        scheme_json = fetch_scheme(scheme_code)
        if scheme_json and "data" in scheme_json:
            st.session_state.df_nav = nav_history_to_df(scheme_json["data"])
            st.session_state.metrics = {}
            st.session_state.pillar_scores = []
            st.session_state.red_flags = []
            st.session_state.mc_results = None
            st.session_state.meta = scheme_json.get("meta", {})
        else:
            st.error("Failed to load NAV data.")

    df_nav = st.session_state.df_nav
    if df_nav is not None and not df_nav.empty:
        # Metrics
        cagr = compute_cagr(df_nav)
        vol = compute_annual_vol(df_nav)
        sharpe = compute_sharpe(cagr, vol, rf=rf_rate)
        st.session_state.metrics = {"cagr": cagr, "vol": vol, "sharpe": sharpe}

        st.subheader("NAV History")
        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(x=df_nav["date"], y=df_nav["nav"], mode="lines", name="NAV"))
        fig_nav.update_layout(height=350, xaxis_title="Date", yaxis_title="NAV")
        st.plotly_chart(fig_nav,use_container_width=True)

        st.subheader("Calculated Metrics")
        m1,m2,m3 = st.columns(3)
        m1.metric("CAGR", f"{cagr:.2%}" if np.isfinite(cagr) else "N/A")
        m2.metric("Volatility", f"{vol:.2%}" if np.isfinite(vol) else "N/A")
        m3.metric("Sharpe (rf {0:.1%})".format(rf_rate), f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")

        # 6-Pillar Scores
        st.subheader("6-Pillar Scoring")
        auto_scores = auto_score_from_metrics(cagr, vol, df_nav, st.session_state.meta)
        PILLARS = [
            ("Return Potential","Expected return vs benchmark"),
            ("Risk Profile","Volatility, downside scenarios"),
            ("Liquidity","Redemption terms, history length"),
            ("Tax Efficiency","Tax treatment / category"),
            ("Manager & Governance","Fund house quality"),
            ("Alignment to Objectives","Investor fit / horizon")
        ]
        cols = st.columns(3)
        pillar_scores = []
        for i,(name,cap) in enumerate(PILLARS):
            col = cols[i%3]
            val = col.slider(name,1,5,value=int(auto_scores[i]),key=f"pillar_{scheme_code}_{i}")
            col.caption(cap)
            pillar_scores.append(val)
        st.session_state.pillar_scores = pillar_scores
        total_score = sum(pillar_scores)
        band = band_from_score(total_score)
        st.markdown(f"**Total Pillar Score:** {total_score}/30")
        st.info(f"Recommendation (before red flags): {band}")

        # Red Flags
        st.subheader("Red Flags")
        red_flags = []
        days = (df_nav["date"].iloc[-1]-df_nav["date"].iloc[0]).days
        if days<365: red_flags.append("Limited track record (<1 yr)")
        if np.isfinite(vol) and vol>0.5: red_flags.append("Very high volatility (>50%)")
        if np.isfinite(cagr) and cagr>1.0: red_flags.append("Unrealistic return (>100%)")
        if df_nav.shape[0]<60: red_flags.append("Few NAV points (<60)")
        manual_flags = ["High fees","No audited NAV","Excessive complexity","Over-concentration"]
        selected_flags = []
        for f in red_flags+manual_flags:
            checked = st.checkbox(f,key=f"flag_{scheme_code}_{f}")
            if checked: selected_flags.append(f)
        st.session_state.red_flags = selected_flags
        final_rec = "Pass (red flags)" if selected_flags else band
        st.success(f"Final Recommendation: {final_rec}")

        # Portfolio
        st.subheader("Portfolio Allocation")
        labels = ["Core","Satellite","Alternatives","Tactical"]
        cols_alloc = st.columns(4)
        raws = []
        for i,c in enumerate(cols_alloc):
            v = c.slider(labels[i],0,100,int(st.session_state.portfolio[i]*100),key=f"alloc_{scheme_code}_{i}")
            raws.append(v)
        total_raw = sum(raws)
        weights = [r/total_raw if total_raw>0 else 0.25 for r in raws]
        st.session_state.portfolio = weights
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=[w*100 for w in weights], hole=0.35)])
        st.plotly_chart(fig_pie,use_container_width=True)

        # Monte Carlo
        st.subheader("Monte Carlo Simulation")
        mc_col1,mc_col2 = st.columns(2)
        with mc_col1:
            sim_years = st.number_input("Years",1,30,value=default_years,key=f"mc_years_{scheme_code}")
            sim_paths = st.number_input("Paths",100,5000,value=default_paths,key=f"mc_paths_{scheme_code}")
            start_val = st.number_input("Start value",1000.0,step=100.0,key=f"mc_start_{scheme_code}")
        with mc_col2:
            use_auto = st.checkbox("Use computed CAGR/Vol",value=True,key=f"mc_use_{scheme_code}")
            mu = cagr if use_auto else st.number_input("Expected return (%)",8.0)/100
            sigma = vol if use_auto else st.number_input("Volatility (%)",12.0)/100

        if st.button("Run Simulation",key=f"mc_run_{scheme_code}"):
            sims = monte_carlo(mu,sigma,int(sim_years),int(sim_paths),start_val)
            st.session_state.mc_results = sims
            median_path = np.median(sims,axis=0)
            run_max = np.maximum.accumulate(median_path)
            mdd = np.max((run_max - median_path)/run_max)
            st.write(f"Median terminal: {median_path[-1]:.2f}, Approx median max drawdown: {mdd:.2%}")

            fig_mc = go.Figure()
            sample_n = min(200,int(sim_paths))
            idx = np.random.choice(int(sim_paths),sample_n,replace=False)
            for i_s in idx:
                fig_mc.add_trace(go.Scatter(x=list(range(int(sim_years)+1)),y=sims[i_s,:],mode="lines",line=dict(width=1),opacity=0.08,showlegend=False))
            fig_mc.add_trace(go.Scatter(x=list(range(int(sim_years)+1)),y=median_path,mode="lines",name="Median",line=dict(width=3)))
            fig_mc.update_layout(height=450,title="Monte Carlo Paths + Median",xaxis_title="Year",yaxis_title="Value")
            st.plotly_chart(fig_mc,use_container_width=True)

else:
    st.info("Enter a fund name and press Search to begin.")
