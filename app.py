import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict
from urllib.parse import urlparse

st.set_page_config(page_title="Indian Mutual Fund Analyzer", layout="wide")

API_SEARCH = "https://api.mfapi.in/mf/search"
API_SCHEME = "https://api.mfapi.in/mf/{}"

# --- Helper functions ---
def universal_scrape(url: str) -> dict:
    """
    Scrape any Indian mutual fund page to detect fund name, and optionally AUM, expense ratio, manager.
    Works for most AMCs by using a browser-like User-Agent.
    """
    info = {"name": None, "AUM": "N/A", "Expense Ratio": "N/A", "Fund Manager": "N/A"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Attempt universal detection of fund name
        if soup.title and soup.title.string:
            info["name"] = soup.title.string.strip()
        else:
            h1 = soup.find("h1")
            if h1:
                info["name"] = h1.text.strip()
        
        # Domain-specific scraping for known AMCs
        domain = urlparse(url).netloc.lower()
        if "bandhanmutual" in domain:
            try:
                info["AUM"] = soup.find("td", text="Assets Under Management").find_next_sibling("td").text.strip()
            except: pass
            try:
                info["Expense Ratio"] = soup.find("td", text="Expense Ratio").find_next_sibling("td").text.strip()
            except: pass
            try:
                info["Fund Manager"] = soup.find("td", text="Fund Manager").find_next_sibling("td").text.strip()
            except: pass
        # You can add more AMCs here with elif blocks

    except requests.exceptions.HTTPError as e:
        st.warning(f"HTTP Error: {e}")
    except Exception as e:
        st.warning(f"Scraping failed: {e}")

    return info

def mf_search(query: str) -> List[Dict]:
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except:
        return []

def fetch_scheme(scheme_code: str) -> Dict:
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except:
        return {}

def nav_history_to_df(nav_json: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(nav_json)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date","nav"]).sort_values("date").reset_index(drop=True)
    return df

def compute_cagr(df: pd.DataFrame) -> float:
    if df.shape[0]<2: return np.nan
    start, end = df["nav"].iloc[0], df["nav"].iloc[-1]
    days = (df["date"].iloc[-1]-df["date"].iloc[0]).days
    if days<=0 or start<=0: return np.nan
    years = days/365.25
    return (end/start)**(1/years)-1

def compute_vol(df: pd.DataFrame) -> float:
    if df.shape[0]<2: return np.nan
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    return df["ret"].std(ddof=0)*np.sqrt(252)

def compute_sharpe(cagr, vol, rf=0.03):
    if not np.isfinite(vol) or vol==0: return np.nan
    return (cagr-rf)/vol

def auto_score(cagr, vol, df):
    scores=[]
    # Return
    if not np.isfinite(cagr): scores.append(3)
    elif cagr>=0.20: scores.append(5)
    elif cagr>=0.12: scores.append(4)
    elif cagr>=0.06: scores.append(3)
    elif cagr>=0.0: scores.append(2)
    else: scores.append(1)
    # Risk
    if not np.isfinite(vol): scores.append(3)
    elif vol<=0.10: scores.append(5)
    elif vol<=0.20: scores.append(4)
    elif vol<=0.35: scores.append(3)
    elif vol<=0.50: scores.append(2)
    else: scores.append(1)
    # Liquidity
    days=(df["date"].iloc[-1]-df["date"].iloc[0]).days if df.shape[0]>=2 else 0
    if days>=3650: scores.append(5)
    elif days>=1825: scores.append(4)
    elif days>=365: scores.append(3)
    elif days>=180: scores.append(2)
    else: scores.append(1)
    # Tax, Manager, Alignment
    scores.extend([3,3,3])
    return scores

def band_from_score(total):
    if total>=25: return "Strong Buy / Anchor allocation"
    if total>=18: return "Opportunistic / Limited allocation"
    return "Pass"

def monte_carlo(mu,sigma,years,n_paths,start_value):
    dt=1.0
    rng=np.random.default_rng()
    rets=rng.normal(loc=mu, scale=sigma, size=(n_paths,years))
    paths=np.empty((n_paths, years+1))
    paths[:,0]=start_value
    for t in range(years):
        paths[:,t+1]=paths[:,t]*(1+rets[:,t])
    return paths

# --- Streamlit UI ---
st.title("Universal Indian Mutual Fund Analyzer")
st.markdown("Paste any Indian mutual fund URL, and the app will fetch NAV, compute metrics, red flags, 6-pillar score, and simulate growth.")

fund_url = st.text_input("Enter fund website URL (any Indian mutual fund)")
rf_rate = st.number_input("Risk-free rate (%)", value=3.0)/100.0
sim_years = st.number_input("Monte Carlo Years",1,30,10)
sim_paths = st.number_input("Monte Carlo Paths",100,5000,1000)
start_val = st.number_input("Starting portfolio value",10000.0,step=100.0)

if st.button("Analyze Fund") and fund_url.strip():
    with st.spinner("Scraping fund info..."):
        scraped = universal_scrape(fund_url)
        fund_name = scraped["name"]
        if not fund_name:
            st.error("Could not detect fund name from URL.")
        else:
            st.success(f"Detected fund name: {fund_name}")
            # MFAPI search
            mf_results = mf_search(fund_name)
            if not mf_results:
                st.error("MFAPI search returned no results for this fund.")
            else:
                best = mf_results[0]
                scheme_code = str(best["schemeCode"])
                scheme_name = best["schemeName"]
                st.info(f"MFAPI matched fund: {scheme_name} (code {scheme_code})")
                scheme_json = fetch_scheme(scheme_code)
                if scheme_json and "data" in scheme_json:
                    df_nav = nav_history_to_df(scheme_json["data"])
                    if df_nav.empty:
                        st.error("NAV history empty.")
                    else:
                        # NAV chart
                        fig=go.Figure()
                        fig.add_trace(go.Scatter(x=df_nav["date"],y=df_nav["nav"],mode="lines",name="NAV"))
                        fig.update_layout(title="NAV History",xaxis_title="Date",yaxis_title="NAV")
                        st.plotly_chart(fig,use_container_width=True)
                        # Metrics
                        cagr=compute_cagr(df_nav)
                        vol=compute_vol(df_nav)
                        sharpe=compute_sharpe(cagr,vol,rf_rate)
                        st.subheader("Metrics")
                        st.metric("CAGR", f"{cagr:.2%}" if np.isfinite(cagr) else "N/A")
                        st.metric("Annual Volatility", f"{vol:.2%}" if np.isfinite(vol) else "N/A")
                        st.metric(f"Sharpe (rf={rf_rate:.1%})", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
                        # Fund info scraped
                        st.subheader("Fund Info from URL")
                        for k,v in scraped.items():
                            st.write(f"**{k}:** {v}")
                        # 6-pillar scoring
                        st.subheader("6-Pillar Score (Auto + Editable)")
                        pillars=["Return Potential","Risk Profile","Liquidity","Tax Efficiency","Manager & Governance","Alignment to Objectives"]
                        auto_scores = auto_score(cagr,vol,df_nav)
                        cols=st.columns(3)
                        pillar_scores=[]
                        for i,p in enumerate(pillars):
                            col=cols[i%3]
                            s=col.slider(p,1,5,int(auto_scores[i]),key=f"pillar_{i}")
                            pillar_scores.append(s)
                        total_score=sum(pillar_scores)
                        band=band_from_score(total_score)
                        st.markdown(f"**Total Score:** {total_score}/30 → {band}")
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
                        run_sim = st.button("Run Simulation")
                        if run_sim:
                            paths=monte_carlo(cagr if np.isfinite(cagr) else 0.08,
                                              vol if np.isfinite(vol) else 0.12,
                                              int(sim_years), int(sim_paths), float(start_val))
                            median_path=np.median(paths,axis=0)
                            fig_mc=go.Figure()
                            x=list(range(int(sim_years)+1))
                            for i_s in range(min(200,int(sim_paths))):
                                fig_mc.add_trace(go.Scatter(x=x,y=paths[i_s,:],mode="lines",line=dict(width=1),opacity=0.08,showlegend=False))
                            fig_mc.add_trace(go.Scatter(x=x,y=median_path,mode="lines",name="Median",line=dict(width=3)))
                            fig_mc.update_layout(title="Monte Carlo Growth",xaxis_title="Year",yaxis_title="Portfolio Value")
                            st.plotly_chart(fig_mc,use_container_width=True)
                            st.write(f"Median terminal value: {median_path[-1]:,.2f}")
                            running_max = np.maximum.accumulate(median_path)
                            drawdowns = (running_max - median_path)/running_max
                            st.write(f"Approx median max drawdown: {np.max(drawdowns):.2%}")
