# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Mutual Fund Evaluator (India)", layout="wide")

API_SEARCH = "https://api.mfapi.in/mf/search"
API_SCHEME = "https://api.mfapi.in/mf/{}"

# ------------------- Helpers -------------------

def mf_search(query: str):
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def fetch_scheme(scheme_code: str):
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch scheme {scheme_code}: {e}")
        return {}

def nav_to_df(nav_json):
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
    cagr = compute_cagr(df)
    vol = compute_vol(df)
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days if df.shape[0]>=2 else 0
    n_points = df.shape[0]

    scores = []

    # Return Potential
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

    # Risk Profile
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

    # Liquidity
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

    # Tax Efficiency
    t_score = 3
    fund_type = meta.get("schemeCategory", "").lower()
    if "equity linked" in fund_type or "elss" in fund_type:
        t_score = 4
    scores.append(t_score)

    # Manager & Governance
    house = meta.get("fund_house", "").lower()
    good_houses = ["bandhan", "hdfc", "icici", "sbi", "aditya birla", "kotak"]
    mgmt_score = 5 if any(h in house for h in good_houses) else 3
    scores.append(mgmt_score)

    # Alignment to Objectives
    scores.append(3)

    return scores, cagr, vol

def band_from_score(total):
    if total >= 25:
        return "Strong Buy / Anchor allocation"
    if total >= 18:
        return "Opportunistic / Limited allocation"
    return "Pass"

def monte_carlo(mu, sigma, years, paths, start):
    rng = np.random.default_rng()
    rets = rng.normal(mu, sigma, size=(paths, years))
    sims = np.empty((paths, years+1))
    sims[:,0] = start
    for t in range(years):
        sims[:,t+1] = sims[:,t]*(1+rets[:,t])
    return sims

# ------------------- Session State Init -------------------

if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_scheme" not in st.session_state:
    st.session_state.selected_scheme = None
if "df_nav" not in st.session_state:
    st.session_state.df_nav = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "pillar_scores" not in st.session_state:
    st.session_state.pillar_scores = [3]*6
if "red_flags" not in st.session_state:
    st.session_state.red_flags = []
if "portfolio" not in st.session_state:
    st.session_state.portfolio = [0.4,0.3,0.2,0.1]
if "mc_results" not in st.session_state:
    st.session_state.mc_results = None

# ------------------- UI -------------------

st.title("Mutual Fund Evaluator (India)")

with st.sidebar:
    st.header("Simulation Settings")
    rf_rate = st.number_input("Risk-free rate (%)", 3.0)/100
    default_years = st.number_input("MC Years", 1, 30, 10)
    default_paths = st.number_input("MC Paths", 100, 5000, 1000)

# Search
st.subheader("Search Mutual Fund")
query = st.text_input("Enter fund name")
search_btn = st.button("Search")

if search_btn and query.strip():
    st.session_state.search_results = mf_search(query.strip())

if st.session_state.search_results:
    options = [f"{r['schemeName']} — {r['schemeCode']}" for r in st.session_state.search_results]
    choice = st.selectbox("Select fund", options, index=0)
    idx = options.index(choice)
    selected = st.session_state.search_results[idx]

    if st.session_state.selected_scheme != selected:
        st.session_state.selected_scheme = selected
        # Fetch NAV
        scheme_json = fetch_scheme(str(selected["schemeCode"]))
        if scheme_json and "data" in scheme_json:
            st.session_state.df_nav = nav_to_df(scheme_json["data"])
            scores, cagr, vol = auto_score(st.session_state.df_nav, scheme_json.get("meta", {}))
            st.session_state.metrics = {"cagr":cagr, "vol":vol, "sharpe":compute_sharpe(cagr,vol,rf_rate)}
            st.session_state.pillar_scores = scores
            # Auto flags
            auto_flags = []
            df = st.session_state.df_nav
            days = (df["date"].iloc[-1]-df["date"].iloc[0]).days
            if days < 365: auto_flags.append("Limited track record (<1yr)")
            if np.isfinite(vol) and vol>0.5: auto_flags.append("High volatility (>50%)")
            if np.isfinite(cagr) and cagr>1.0: auto_flags.append("Unrealistic CAGR (>100%)")
            if df.shape[0]<60: auto_flags.append("Few NAV points (<60)")
            st.session_state.red_flags = auto_flags
            # Portfolio default
            st.session_state.portfolio = [0.4,0.3,0.2,0.1]

# If a scheme is selected
if st.session_state.selected_scheme and st.session_state.df_nav is not None:
    df = st.session_state.df_nav
    st.subheader(f"NAV History: {st.session_state.selected_scheme['schemeName']}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["nav"], mode="lines", name="NAV"))
    fig.update_layout(height=350, xaxis_title="Date", yaxis_title="NAV")
    st.plotly_chart(fig,use_container_width=True)

    # Metrics
    st.subheader("Calculated Metrics")
    c1,c2,c3 = st.columns(3)
    c1.metric("CAGR", f"{st.session_state.metrics['cagr']:.2%}" if np.isfinite(st.session_state.metrics['cagr']) else "N/A")
    c2.metric("Volatility", f"{st.session_state.metrics['vol']:.2%}" if np.isfinite(st.session_state.metrics['vol']) else "N/A")
    c3.metric("Sharpe", f"{st.session_state.metrics['sharpe']:.2f}" if np.isfinite(st.session_state.metrics['sharpe']) else "N/A")

    # Pillars
    st.subheader("6-Pillar Scoring")
    PILLARS = [
        ("Return Potential", "Expected return vs benchmark, consistency"),
        ("Risk Profile", "Volatility, concentration, downside"),
        ("Liquidity", "NAV history length & fund size"),
        ("Tax Efficiency", "Based on fund type"),
        ("Manager & Governance", "Fund house reputation"),
        ("Alignment to Objectives", "Fit with investor goals")
    ]
    cols = st.columns(3)
    for i,(name,caption) in enumerate(PILLARS):
        s = cols[i%3].slider(name,1,5,value=st.session_state.pillar_scores[i],key=f"p_{i}")
        cols[i%3].caption(caption)
        st.session_state.pillar_scores[i] = s
    total = sum(st.session_state.pillar_scores)
    band = band_from_score(total)
    st.write(f"**Total Pillar Score:** {total}/30")
    st.info(f"Recommendation (pre-red flags): {band}")

    # Red Flags
    st.subheader("Red Flags")
    st.write("Auto + manual flags")
    manual_flags = ["High fees","No audited NAV","Complex structure","Over-concentration","High exit load"]
    for i,f in enumerate(manual_flags):
        if st.checkbox(f, value=f in st.session_state.red_flags, key=f"rf_{i}"):
            if f not in st.session_state.red_flags:
                st.session_state.red_flags.append(f)
        else:
            if f in st.session_state.red_flags:
                st.session_state.red_flags.remove(f)
    if st.session_state.red_flags:
        st.write("**Active Flags:**")
        for f in st.session_state.red_flags:
            st.write(f"- {f}")
        final_rec = "Pass (red flags)"
    else:
        final_rec = band
    st.success(f"Final Recommendation: {final_rec}")

    # Portfolio
    st.subheader("Portfolio Allocation")
    labels = ["Core","Satellite","Alternatives","Tactical"]
    cols_alloc = st.columns(4)
    raws = []
    for i,c in enumerate(cols_alloc):
        v = c.slider(labels[i],0,100,int(st.session_state.portfolio[i]*100),key=f"alloc_{i}")
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
        sim_years = st.number_input("Years",1,30,value=default_years,key="mc_years")
        sim_paths = st.number_input("Paths",100,5000,value=default_paths,key="mc_paths")
        start_val = st.number_input("Start value",1000.0,step=100.0,key="mc_start")
    with mc_col2:
        use_auto = st.checkbox("Use computed CAGR/Vol",value=True,key="mc_use")
        mu = st.session_state.metrics["cagr"] if use_auto else st.number_input("Expected return (%)",8.0)/100
        sigma = st.session_state.metrics["vol"] if use_auto else st.number_input("Volatility (%)",12.0)/100

    if st.button("Run Simulation"):
        sims = monte_carlo(mu,sigma,int(sim_years),int(sim_paths),start_val)
        st.session_state.mc_results = sims
        median_path = np.median(sims,axis=0)
        run_max = np.maximum.accumulate(median_path)
        mdd = np.max((run_max - median_path)/run_max)
        st.write(f"Median terminal: {median_path[-1]:.2f}, Approx median max drawdown: {mdd:.2%}")

        # Plot sample
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
