import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from typing import List, Dict
from datetime import datetime
from urllib.parse import urlparse

st.set_page_config(page_title="Indian Mutual Fund Analyzer", layout="wide")

API_SEARCH = "https://api.mfapi.in/mf/search"
API_SCHEME = "https://api.mfapi.in/mf/{}"

# --------------------- Helper Functions ---------------------

# --- MFAPI Functions ---
def mf_search(query: str) -> List[Dict]:
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"MFAPI search failed: {e}")
        return []

def fetch_scheme(scheme_code: str) -> Dict:
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Failed to fetch scheme {scheme_code}: {e}")
        return {}

def nav_history_to_df(nav_json: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(nav_json)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date","nav"]).sort_values("date").reset_index(drop=True)
    return df

# --- Metrics ---
def compute_cagr(df: pd.DataFrame) -> float:
    if df.shape[0]<2: return float("nan")
    start, end = df["nav"].iloc[0], df["nav"].iloc[-1]
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    if days<=0 or start<=0: return float("nan")
    years = days/365.25
    return (end/start)**(1/years)-1

def compute_annual_vol(df: pd.DataFrame) -> float:
    if df.shape[0]<2: return float("nan")
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    return df["ret"].std(ddof=0)*np.sqrt(252)

def compute_sharpe(cagr: float, vol: float, rf: float=0.03) -> float:
    if not np.isfinite(vol) or vol==0: return float("nan")
    return (cagr - rf)/vol

def auto_score_from_metrics(cagr: float, vol: float, df: pd.DataFrame) -> List[int]:
    scores=[]
    # Return Potential
    if not np.isfinite(cagr): scores.append(3)
    elif cagr>=0.20: scores.append(5)
    elif cagr>=0.12: scores.append(4)
    elif cagr>=0.06: scores.append(3)
    elif cagr>=0.0: scores.append(2)
    else: scores.append(1)
    # Risk Profile
    if not np.isfinite(vol): scores.append(3)
    elif vol<=0.10: scores.append(5)
    elif vol<=0.20: scores.append(4)
    elif vol<=0.35: scores.append(3)
    elif vol<=0.50: scores.append(2)
    else: scores.append(1)
    # Liquidity (heuristic)
    days=(df["date"].iloc[-1]-df["date"].iloc[0]).days if df.shape[0]>=2 else 0
    if days>=3650: scores.append(5)
    elif days>=1825: scores.append(4)
    elif days>=365: scores.append(3)
    elif days>=180: scores.append(2)
    else: scores.append(1)
    # Others default neutral
    scores.extend([3,3,3])
    return scores

def band_from_score(total:int)->str:
    if total>=25: return "Strong Buy / Anchor allocation"
    if total>=18: return "Opportunistic / Limited allocation"
    return "Pass"

def monte_carlo(mu:float,sigma:float,years:int,n_paths:int,start_value:float):
    dt=1.0
    rng=np.random.default_rng()
    steps=years
    rets=rng.normal(loc=mu,scale=sigma,size=(n_paths,steps))
    paths=np.empty((n_paths,steps+1),dtype=float)
    paths[:,0]=start_value
    for t in range(steps):
        paths[:,t+1]=paths[:,t]*(1+rets[:,t])
    return paths

# --- Scraper Function ---
def universal_scrape(url:str)->dict:
    info={"name":None,"AUM":"N/A","Expense Ratio":"N/A","Fund Manager":"N/A"}
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
    try:
        resp=requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup=BeautifulSoup(resp.text,"html.parser")
        # Try h1 first
        h1=soup.find("h1")
        if h1 and h1.text.strip():
            info["name"]=h1.text.strip()
        # domain-specific Bandhan
        domain=urlparse(url).netloc.lower()
        if "bandhanmutual" in domain:
            try: info["AUM"]=soup.find("td",text="Assets Under Management").find_next_sibling("td").text.strip()
            except: pass
            try: info["Expense Ratio"]=soup.find("td",text="Expense Ratio").find_next_sibling("td").text.strip()
            except: pass
            try: info["Fund Manager"]=soup.find("td",text="Fund Manager").find_next_sibling("td").text.strip()
            except: pass
    except Exception as e:
        st.warning(f"Scraping failed: {e}")
    return info

# --------------------- UI ---------------------
st.title("Indian Mutual Fund Analyzer — MFAPI + Website Details")
st.markdown("Enter **fund name** to fetch NAV & metrics and/or paste **fund website URL** for additional info.")

col1, col2 = st.columns(2)
with col1:
    fund_name_input = st.text_input("Enter Fund Name (MFAPI search)")
with col2:
    fund_url_input = st.text_input("Enter Fund Website URL")

rf_rate = st.number_input("Risk-free rate (%)", value=3.0, step=0.1)/100.0

# --------------------- Scrape Website ---------------------
scraped_info = None
if fund_url_input.strip():
    with st.spinner("Scraping website..."):
        scraped_info = universal_scrape(fund_url_input)
        st.subheader("Website Info")
        st.json(scraped_info)

# --------------------- MFAPI Fetch ---------------------
if fund_name_input.strip():
    with st.spinner("Searching MFAPI..."):
        search_results = mf_search(fund_name_input.strip())
    if not search_results:
        st.warning("No MFAPI results found.")
    else:
        # pick first result
        scheme_code = str(search_results[0]["schemeCode"])
        scheme_name = search_results[0]["schemeName"]
        st.success(f"MFAPI: {scheme_name} (code: {scheme_code})")
        with st.spinner("Fetching NAV history..."):
            scheme_json = fetch_scheme(scheme_code)
        if scheme_json and "data" in scheme_json:
            df_nav = nav_history_to_df(scheme_json["data"])
            if df_nav.empty:
                st.error("NAV history empty.")
            else:
                st.subheader("NAV History")
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=df_nav["date"],y=df_nav["nav"],mode="lines"))
                st.plotly_chart(fig,use_container_width=True)

                # Metrics
                cagr = compute_cagr(df_nav)
                vol = compute_annual_vol(df_nav)
                sharpe = compute_sharpe(cagr, vol, rf_rate)
                st.subheader("Calculated Metrics")
                st.metric("CAGR", f"{cagr:.2%}" if np.isfinite(cagr) else "N/A")
                st.metric("Annualized Volatility", f"{vol:.2%}" if np.isfinite(vol) else "N/A")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")

                # Auto 6-pillar scoring
                auto_scores = auto_score_from_metrics(cagr, vol, df_nav)
                PILLARS = ["Return Potential","Risk Profile","Liquidity","Tax Efficiency","Manager & Governance","Alignment to Objectives"]
                st.subheader("6-Pillar Auto Scores")
                cols = st.columns(3)
                pillar_scores=[]
                for i,name in enumerate(PILLARS):
                    col=cols[i%3]
                    s=col.slider(name,1,5,value=int(auto_scores[i]),key=f"pillar_{i}")
                    pillar_scores.append(s)
                total_score=sum(pillar_scores)
                band=band_from_score(total_score)
                st.markdown(f"**Total Pillar Score:** {total_score}/30")
                st.info(f"Recommendation: {band}")

                # Monte Carlo
                st.subheader("Monte Carlo Simulation")
                sim_years = st.number_input("Years to simulate", 1,30,10)
                sim_paths = st.number_input("Number of paths", 100,5000,1000)
                start_val = st.number_input("Starting value", 10000.0, step=100.0)
                run_sim = st.button("Run Monte Carlo")
                if run_sim:
                    mu = cagr if np.isfinite(cagr) else 0.08
                    sigma = vol if np.isfinite(vol) else 0.12
                    paths = monte_carlo(mu,sigma,int(sim_years),int(sim_paths),float(start_val))
                    median_path=np.median(paths,axis=0)
                    fig_mc=go.Figure()
                    x=list(range(int(sim_years)+1))
                    for i_s in range(min(200,int(sim_paths))):
                        fig_mc.add_trace(go.Scatter(x=x,y=paths[i_s,:],mode="lines",
                                                    line=dict(width=1),opacity=0.08,showlegend=False))
                    fig_mc.add_trace(go.Scatter(x=x,y=median_path,mode="lines",name="Median",line=dict(width=3)))
                    fig_mc.update_layout(title="Monte Carlo Growth",xaxis_title="Year",yaxis_title="Portfolio Value")
                    st.plotly_chart(fig_mc,use_container_width=True)
                    st.write(f"Median terminal value: {median_path[-1]:,.2f}")
                    running_max = np.maximum.accumulate(median_path)
                    drawdowns = (running_max - median_path)/running_max
                    st.write(f"Approx median max drawdown: {np.max(drawdowns):.2%}")
