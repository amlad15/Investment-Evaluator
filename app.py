import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Mutual Fund Analyzer", layout="wide")

API_SEARCH = "https://api.mfapi.in/mf/search"
API_SCHEME = "https://api.mfapi.in/mf/{}"

# --- Helper functions ---
def mf_search(query: str) -> List[Dict]:
    """Search MFAPI for funds"""
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def fetch_scheme(scheme_code: str) -> Dict:
    """Fetch scheme details from MFAPI"""
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Scheme fetch failed: {e}")
        return {}

def nav_history_to_df(nav_json: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(nav_json)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    return df

def compute_cagr(df: pd.DataFrame) -> float:
    if df.shape[0] < 2:
        return np.nan
    start = df["nav"].iloc[0]
    end = df["nav"].iloc[-1]
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    if days <= 0 or start <= 0:
        return np.nan
    years = days / 365.25
    return (end / start) ** (1/years) - 1

def compute_vol(df: pd.DataFrame) -> float:
    if df.shape[0] < 2:
        return np.nan
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    return df["ret"].std(ddof=0) * np.sqrt(252)

def compute_sharpe(cagr: float, vol: float, rf: float = 0.03) -> float:
    if not np.isfinite(vol) or vol==0:
        return np.nan
    return (cagr - rf) / vol

def scrape_bandhan_fund(url: str) -> Dict:
    """Scrape Bandhan mutual fund page for AUM, expense ratio, manager"""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        info = {}
        # Example selectors (may change if site updates)
        try:
            info["AUM"] = soup.find("td", text="Assets Under Management").find_next_sibling("td").text.strip()
        except: info["AUM"] = "N/A"
        try:
            info["Expense Ratio"] = soup.find("td", text="Expense Ratio").find_next_sibling("td").text.strip()
        except: info["Expense Ratio"] = "N/A"
        try:
            info["Fund Manager"] = soup.find("td", text="Fund Manager").find_next_sibling("td").text.strip()
        except: info["Fund Manager"] = "N/A"
        return info
    except Exception as e:
        st.warning(f"Scraping failed: {e}")
        return {"AUM":"N/A","Expense Ratio":"N/A","Fund Manager":"N/A"}

def auto_score(cagr, vol, df):
    scores = []
    # Return Potential
    if not np.isfinite(cagr): scores.append(3)
    elif cagr>=0.20: scores.append(5)
    elif cagr>=0.12: scores.append(4)
    elif cagr>=0.06: scores.append(3)
    elif cagr>=0.0: scores.append(2)
    else: scores.append(1)
    # Risk Profile (lower vol better)
    if not np.isfinite(vol): scores.append(3)
    elif vol<=0.10: scores.append(5)
    elif vol<=0.20: scores.append(4)
    elif vol<=0.35: scores.append(3)
    elif vol<=0.50: scores.append(2)
    else: scores.append(1)
    # Liquidity: history length heuristic
    days = (df["date"].iloc[-1]-df["date"].iloc[0]).days if df.shape[0]>=2 else 0
    if days>=3650: scores.append(5)
    elif days>=1825: scores.append(4)
    elif days>=365: scores.append(3)
    elif days>=180: scores.append(2)
    else: scores.append(1)
    # Tax, Manager, Alignment: defaults
    scores.extend([3,3,3])
    return scores

def monte_carlo(mu,sigma,years,n_paths,start_value):
    dt=1.0
    rng=np.random.default_rng()
    rets=rng.normal(loc=mu, scale=sigma, size=(n_paths,years))
    paths=np.empty((n_paths, years+1))
    paths[:,0]=start_value
    for t in range(years):
        paths[:,t+1]=paths[:,t]*(1+rets[:,t])
    return paths

def band_from_score(total):
    if total>=25: return "Strong Buy / Anchor allocation"
    if total>=18: return "Opportunistic / Limited allocation"
    return "Pass"

# --- Streamlit UI ---
st.title("Mutual Fund Analyzer (India)")

st.markdown("Search Indian mutual funds, see metrics, red flags, auto 6-pillar score, and simulate growth.")

query = st.text_input("Enter fund name (e.g., Bandhan Small Cap)")
search_btn = st.button("Search")

if search_btn and query.strip():
    with st.spinner("Searching funds..."):
        results = mf_search(query.strip())
    if results:
        options = [f"{r['schemeName']} ({r['schemeCode']})" for r in results]
        choice = st.selectbox("Select a scheme", options)
        idx = options.index(choice)
        scheme_code = str(results[idx]["schemeCode"])
        scheme_name = results[idx]["schemeName"]
        st.success(f"Selected: {scheme_name}")
        
        # Fetch NAV
        with st.spinner("Fetching NAV..."):
            scheme_json = fetch_scheme(scheme_code)
        if scheme_json and "data" in scheme_json:
            df_nav = nav_history_to_df(scheme_json["data"])
            if not df_nav.empty:
                # Show NAV chart
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=df_nav["date"],y=df_nav["nav"],mode="lines",name="NAV"))
                fig.update_layout(title="NAV History",xaxis_title="Date",yaxis_title="NAV")
                st.plotly_chart(fig,use_container_width=True)
                
                # Metrics
                cagr=compute_cagr(df_nav)
                vol=compute_vol(df_nav)
                sharpe=compute_sharpe(cagr,vol)
                
                st.subheader("Metrics")
                st.metric("CAGR", f"{cagr:.2%}" if np.isfinite(cagr) else "N/A")
                st.metric("Annual Volatility", f"{vol:.2%}" if np.isfinite(vol) else "N/A")
                st.metric("Sharpe (rf=3%)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
                
                # Scrape Bandhan page (example)
                st.subheader("Fund Info (scraped from Bandhan MF site)")
                fund_url = st.text_input("Enter Bandhan fund URL", value="https://bandhanmutual.com/mutual-funds/equity-funds/bandhan-small-cap-fund/regular")
                if fund_url:
                    scraped_info = scrape_bandhan_fund(fund_url)
                    for k,v in scraped_info.items():
                        st.write(f"**{k}:** {v}")
                
                # Auto 6-pillar score
                st.subheader("6-Pillar Score")
                pillars=["Return Potential","Risk Profile","Liquidity","Tax Efficiency","Manager & Governance","Alignment to Objectives"]
                auto_scores=auto_score(cagr,vol,df_nav)
                pillar_scores=[]
                cols=st.columns(3)
                for i,p in enumerate(pillars):
                    col=cols[i%3]
                    s=col.slider(p,1,5,int(auto_scores[i]),key=f"pillar_{i}")
                    pillar_scores.append(s)
                total_score=sum(pillar_scores)
                band=band_from_score(total_score)
                st.markdown(f"**Total Score:** {total_score}/30 — {band}")
                
                # Red flags
                st.subheader("Red Flags")
                flags=[]
                days=(df_nav["date"].iloc[-1]-df_nav["date"].iloc[0]).days
                if days<365: flags.append("Short history (<1 year)")
                if np.isfinite(vol) and vol>0.50: flags.append("Very high volatility")
                if np.isfinite(cagr) and cagr>1.0: flags.append("Unrealistic CAGR")
                st.write(flags if flags else "No automatic red flags detected")
                
                # Monte Carlo
                st.subheader("Monte Carlo Simulation")
                sim_years=st.number_input("Years to simulate",1,30,10)
                sim_paths=st.number_input("Number of paths",100,5000,1000)
                start_val=st.number_input("Start value",10000.0,step=100.0)
                run_sim=st.button("Run Simulation")
                if run_sim:
                    mu=cagr if np.isfinite(cagr) else 0.08
                    sigma=vol if np.isfinite(vol) else 0.12
                    with st.spinner("Simulating..."):
                        paths=monte_carlo(mu,sigma,int(sim_years),int(sim_paths),start_val)
                        median_path=np.median(paths,axis=0)
                        fig_mc=go.Figure()
                        x=list(range(int(sim_years)+1))
                        for i_p in range(min(200,int(sim_paths))):
                            fig_mc.add_trace(go.Scatter(x=x,y=paths[i_p,:],mode="lines",line=dict(width=1),opacity=0.05,showlegend=False))
                        fig_mc.add_trace(go.Scatter(x=x,y=median_path,mode="lines",name="Median",line=dict(width=3)))
                        fig_mc.update_layout(title="Monte Carlo Simulation",xaxis_title="Year",yaxis_title="Value")
                        st.plotly_chart(fig_mc,use_container_width=True)
            else:
                st.error("NAV data empty")
        else:
            st.error("Scheme fetch failed")
    else:
        st.warning("No results found")
