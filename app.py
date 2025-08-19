# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Evaluator (India)", layout="wide")

API_SEARCH = "https://api.mfapi.in/mf/search"
API_SCHEME = "https://api.mfapi.in/mf/{}"

# ------------------- Helpers -------------------

def mf_search(query: str):
    """Search MFAPI for fund schemes."""
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def fetch_scheme(scheme_code: str):
    """Fetch scheme data (NAV history + meta)."""
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch scheme {scheme_code}: {e}")
        return {}

def nav_to_df(nav_json):
    """Convert MFAPI NAV data to DataFrame."""
    df = pd.DataFrame(nav_json)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    return df

def compute_cagr(df):
    if df.shape[0] < 2:
        return np.nan
    start = df["nav"].iloc[0]
    end = df["nav"].iloc[-1]
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    if days <= 0 or start <= 0:
        return np.nan
    years = days / 365.25
    return (end / start) ** (1 / years) - 1

def compute_vol(df):
    if df.shape[0] < 2:
        return np.nan
    df["ret"] = df["nav"].pct_change()
    return df["ret"].std(ddof=0) * np.sqrt(252)

def compute_sharpe(cagr, vol, rf=0.03):
    if not np.isfinite(vol) or vol == 0:
        return np.nan
    return (cagr - rf) / vol

def auto_score(df, meta):
    """Dynamic 6-pillar scoring based on NAV & meta."""
    cagr = compute_cagr(df)
    vol = compute_vol(df)
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days if df.shape[0]>=2 else 0
    n_points = df.shape[0]

    scores = []

    # 1. Return Potential (CAGR)
    if not np.isfinite(cagr):
        scores.append(3)
    elif cagr >= 0.25:
        scores.append(5)
    elif cagr >= 0.18:
        scores.append(4)
    elif cagr >= 0.12:
        scores.append(3)
    elif cagr >= 0.06:
        scores.append(2)
    else:
        scores.append(1)

    # 2. Risk Profile (inverse volatility)
    if not np.isfinite(vol):
        scores.append(3)
    elif vol <= 0.10:
        scores.append(5)
    elif vol <= 0.15:
        scores.append(4)
    elif vol <= 0.25:
        scores.append(3)
    elif vol <= 0.40:
        scores.append(2)
    else:
        scores.append(1)

    # 3. Liquidity (based on history length + points)
    if days >= 3650 and n_points >= 1200:
        scores.append(5)
    elif days >= 1825 and n_points >= 600:
        scores.append(4)
    elif days >= 365 and n_points >= 250:
        scores.append(3)
    elif days >= 180:
        scores.append(2)
    else:
        scores.append(1)

    # 4. Tax Efficiency (heuristic: default 3, reduce if fundType is Equity or ELSS)
    t_score = 3
    fund_type = meta.get("schemeCategory", "").lower()
    if "equity linked" in fund_type or "elss" in fund_type:
        t_score = 4
    scores.append(t_score)

    # 5. Manager & Governance (heuristic based on fund house reputation, simple mapping)
    house = meta.get("fund_house", "").lower()
    good_houses = ["bandhan", "hdfc", "icici", "sbi", "aditya birla", "kotak"]
    mgmt_score = 5 if any(h in house for h in good_houses) else 3
    scores.append(mgmt_score)

    # 6. Alignment to Objectives (default neutral 3, user can override)
    scores.append(3)

    return scores, cagr, vol

def band_from_score(total):
    if total >= 25:
        return "Strong Buy / Anchor allocation"
    if total >= 18:
        return "Opportunistic / Limited allocation"
    return "Pass"

def monte_carlo(mu, sigma, years, paths, start):
    dt = 1.0
    rng = np.random.default_rng()
    rets = rng.normal(mu, sigma, size=(paths, years))
    sims = np.empty((paths, years+1))
    sims[:,0] = start
    for t in range(years):
        sims[:,t+1] = sims[:,t]*(1+rets[:,t])
    return sims

# ------------------- UI -------------------

st.title("Mutual Fund Evaluator (India)")

with st.sidebar:
    st.header("Simulation Settings")
    rf_rate = st.number_input("Risk-free rate (%)", 3.0)/100
    default_years = st.number_input("MC Years", 1, 30, 10)
    default_paths = st.number_input("MC Paths", 100, 5000, 1000)

# 1) Search
st.subheader("Search Mutual Fund")
query = st.text_input("Enter fund name")
search_btn = st.button("Search")

search_results = []
if search_btn and query.strip():
    with st.spinner("Searching MFAPI..."):
        search_results = mf_search(query.strip())
    if not search_results:
        st.warning("No results found")
if search_results:
    options = [f"{r['schemeName']} — {r['schemeCode']}" for r in search_results]
    choice = st.selectbox("Select fund", options)
    idx = options.index(choice)
    selected = search_results[idx]
    scheme_code = str(selected["schemeCode"])
    scheme_name = selected["schemeName"]
    st.success(f"Selected: {scheme_name} ({scheme_code})")
    
    with st.spinner("Fetching NAV history..."):
        scheme_json = fetch_scheme(scheme_code)

    if scheme_json and "data" in scheme_json:
        df_nav = nav_to_df(scheme_json["data"])
        if df_nav.empty:
            st.error("NAV data empty")
        else:
            # Display NAV
            st.subheader("NAV History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_nav["date"], y=df_nav["nav"], mode="lines", name="NAV"))
            fig.update_layout(height=350, xaxis_title="Date", yaxis_title="NAV")
            st.plotly_chart(fig, use_container_width=True)

            # Metrics & Auto-score
            scores, cagr, vol = auto_score(df_nav, scheme_json.get("meta", {}))
            st.subheader("Calculated Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("CAGR", f"{cagr:.2%}" if np.isfinite(cagr) else "N/A")
            c2.metric("Volatility", f"{vol:.2%}" if np.isfinite(vol) else "N/A")
            c3.metric("Sharpe", f"{compute_sharpe(cagr, vol, rf_rate):.2f}" if np.isfinite(cagr) else "N/A")

            # 6-Pillar sliders
            st.subheader("6-Pillar Scoring")
            PILLARS = [
                ("Return Potential", "Expected return vs benchmark, consistency of returns"),
                ("Risk Profile", "Volatility, concentration, downside"),
                ("Liquidity", "NAV history length and fund size"),
                ("Tax Efficiency", "Estimated tax efficiency based on fund type"),
                ("Manager & Governance", "Fund house reputation and track record"),
                ("Alignment to Objectives", "Fit with investor goals (manual override)")
            ]
            cols = st.columns(3)
            pillar_scores = []
            for i, (name, caption) in enumerate(PILLARS):
                col = cols[i%3]
                s = col.slider(name, 1,5, int(scores[i]), key=f"p_{i}")
                col.caption(caption)
                pillar_scores.append(s)
            total = sum(pillar_scores)
            band = band_from_score(total)
            st.write(f"**Total Pillar Score:** {total}/30")
            st.info(f"Recommendation (pre-red flags): {band}")

            # Red flags
            st.subheader("Red Flags")
            auto_flags = []
            days = (df_nav["date"].iloc[-1]-df_nav["date"].iloc[0]).days
            if days < 365:
                auto_flags.append("Limited track record (<1yr)")
            if np.isfinite(vol) and vol > 0.50:
                auto_flags.append("High volatility (>50%)")
            if np.isfinite(cagr) and cagr > 1.0:
                auto_flags.append("Unrealistic CAGR (>100%)")
            if df_nav.shape[0] < 60:
                auto_flags.append("Few NAV points (<60)")

            manual_flags = [
                "High fees", "No audited NAV", "Complex structure", "Over-concentration", "High exit load"
            ]

            selected_flags = auto_flags.copy()
            if auto_flags:
                st.warning("Auto-detected flags:")
                for f in auto_flags:
                    st.write(f"- {f}")

            st.write("Manual flags (check if applicable):")
            cols_flags = st.columns(2)
            for i, f in enumerate(manual_flags):
                if cols_flags[i%2].checkbox(f, key=f"mf_{i}"):
                    selected_flags.append(f)
            selected_flags = list(dict.fromkeys(selected_flags))

            if selected_flags:
                st.write("**Active flags:**")
                for f in selected_flags:
                    st.write(f"- {f}")

            final_rec = "Pass (red flag)" if selected_flags else band
            st.success(f"Final recommendation: {final_rec}")

            # Portfolio Construction
            st.markdown("---")
            st.subheader("Portfolio Allocation")
            c1,c2,c3,c4 = st.columns(4)
            raw = np.array([
                c1.slider("Core",0,100,40,key="r_core"),
                c2.slider("Satellite",0,100,30,key="r_sat"),
                c3.slider("Alternatives",0,100,20,key="r_alt"),
                c4.slider("Tactical",0,100,10,key="r_tac")
            ],dtype=float)
            weights = raw/raw.sum() if raw.sum()>0 else np.array([0.25]*4)
            labels = ["Core","Satellite","Alternatives","Tactical"]
            fig_pie = go.Figure(data=[go.Pie(labels=labels,values=weights*100,hole=0.35)])
            st.plotly_chart(fig_pie, use_container_width=True)

            # Monte Carlo
            st.markdown("---")
            st.subheader("Monte Carlo Simulation")
            sim_years = st.number_input("Years",1,30,value=default_years)
            sim_paths = st.number_input("Paths",100,5000,value=default_paths)
            start_val = st.number_input("Start portfolio value",value=10000.0)
            use_auto = st.checkbox("Use auto CAGR & Volatility",value=True)
            if not use_auto:
                mu = st.number_input("Expected return (%)",8.0)/100
                sigma = st.number_input("Volatility (%)",12.0)/100
            else:
                mu = cagr if np.isfinite(cagr) else 0.08
                sigma = vol if np.isfinite(vol) else 0.12
            run_mc = st.button("Run Monte Carlo")
            if run_mc:
                sims = monte_carlo(mu,sigma,int(sim_years),int(sim_paths),start_val)
                median = np.median(sims,axis=0)
                run_max = np.maximum.accumulate(median)
                mdd = np.max((run_max-median)/run_max)
                st.write(f"Median terminal value: {median[-1]:,.2f}")
                st.write(f"Approx median max drawdown: {mdd:.2%}")
                # plot sample paths
                fig_mc = go.Figure()
                sample_n = min(200,int(sim_paths))
                idx = np.random.choice(int(sim_paths),sample_n,replace=False)
                x = list(range(int(sim_years)+1))
                for i_s in idx:
                    fig_mc.add_trace(go.Scatter(x=x,y=sims[i_s,:],mode="lines",line=dict(width=1),opacity=0.08,showlegend=False))
                fig_mc.add_trace(go.Scatter(x=x,y=median,mode="lines",name="Median",line=dict(width=3)))
                fig_mc.update_layout(title="Monte Carlo Simulation", xaxis_title="Year",yaxis_title="Portfolio Value",height=450)
                st.plotly_chart(fig_mc,use_container_width=True)

else:
    st.info("Enter fund name and press Search to start.")
