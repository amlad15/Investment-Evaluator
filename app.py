# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Mutual Fund Quick Analyzer", layout="wide")
st.title("Mutual Fund Quick Analyzer (India)")
st.markdown(
    "Search a fund, view NAV, returns, volatility, Sharpe ratio, rolling returns, drawdowns, and simple red flags."
)

API_SEARCH = "https://api.mfapi.in/mf/search"
API_SCHEME = "https://api.mfapi.in/mf/{}"

# --- Helper functions ---
def mf_search(query):
    try:
        resp = requests.get(API_SEARCH, params={"q": query}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return []

def fetch_scheme(scheme_code):
    try:
        resp = requests.get(API_SCHEME.format(scheme_code), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return {}

def nav_to_df(nav_json):
    df = pd.DataFrame(nav_json)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date","nav"]).sort_values("date").reset_index(drop=True)
    return df

def compute_cagr(df):
    if len(df)<2: return np.nan
    start, end = df["nav"].iloc[0], df["nav"].iloc[-1]
    days = (df["date"].iloc[-1]-df["date"].iloc[0]).days
    if days<=0 or start<=0: return np.nan
    years = days/365.25
    return (end/start)**(1/years)-1

def compute_vol(df):
    if len(df)<2: return np.nan
    df["ret"] = df["nav"].pct_change()
    return df["ret"].std(ddof=0)*np.sqrt(252)

def compute_sharpe(cagr, vol, rf=0.03):
    if not np.isfinite(vol) or vol==0: return np.nan
    return (cagr - rf)/vol

def rolling_returns(df, window_days=252):
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    df["rolling"] = df["ret"].rolling(window=window_days).apply(lambda x: np.prod(1+x)-1, raw=True)
    return df

def drawdown(df):
    df = df.copy()
    df["cum_max"] = df["nav"].cummax()
    df["drawdown"] = (df["cum_max"] - df["nav"])/df["cum_max"]
    return df

# --- Search ---
query = st.text_input("Enter mutual fund name (e.g., 'Bandhan Small Cap')", "")
if st.button("Search"):
    results = mf_search(query)
    if not results:
        st.warning("No results found.")
    else:
        options = [f"{r['schemeName']} — {r['schemeCode']}" for r in results]
        choice = st.selectbox("Select a fund", options)
        idx = options.index(choice)
        scheme_code = results[idx]["schemeCode"]
        scheme_name = results[idx]["schemeName"]

        # Fetch NAV
        data = fetch_scheme(scheme_code)
        if "data" not in data:
            st.error("Failed to fetch NAV data.")
        else:
            df_nav = nav_to_df(data["data"])
            if df_nav.empty:
                st.error("NAV data empty.")
            else:
                # Display basic info
                st.subheader(f"{scheme_name} — NAV History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_nav["date"], y=df_nav["nav"], mode="lines", name="NAV"))
                fig.update_layout(height=350, xaxis_title="Date", yaxis_title="NAV")
                st.plotly_chart(fig,use_container_width=True)

                # Compute metrics
                cagr = compute_cagr(df_nav)
                vol = compute_vol(df_nav)
                sharpe = compute_sharpe(cagr, vol)

                st.subheader("Key Metrics")
                col1,col2,col3 = st.columns(3)
                col1.metric("CAGR", f"{cagr:.2%}" if np.isfinite(cagr) else "N/A")
                col2.metric("Annualized Volatility", f"{vol:.2%}" if np.isfinite(vol) else "N/A")
                col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")

                # Red flags
                st.subheader("Red Flags")
                flags = []
                days = (df_nav["date"].iloc[-1]-df_nav["date"].iloc[0]).days
                if days<365: flags.append("Limited track record (<1 year)")
                if np.isfinite(vol) and vol>0.5: flags.append("Very high volatility (>50%)")
                if len(df_nav)<60: flags.append("Few NAV points (<60)")
                if flags:
                    for f in flags: st.warning(f)
                else:
                    st.success("No obvious red flags detected.")

                # Rolling returns
                st.subheader("Rolling 1-Year Returns")
                df_roll = rolling_returns(df_nav, window_days=252)
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=df_roll["date"], y=df_roll["rolling"], mode="lines", name="1Y Rolling Return"))
                fig_roll.update_layout(height=350, xaxis_title="Date", yaxis_title="Return", yaxis_tickformat=".2%")
                st.plotly_chart(fig_roll,use_container_width=True)

                # Drawdowns
                st.subheader("Drawdown")
                df_dd = drawdown(df_nav)
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=df_dd["date"], y=df_dd["drawdown"], mode="lines", name="Drawdown"))
                fig_dd.update_layout(height=350, xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".2%")
                st.plotly_chart(fig_dd,use_container_width=True)
