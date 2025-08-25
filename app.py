# app.py — TruHome Monthly Insights (Investor View)
# Clean, simple, and robust. Upload a CSV and get instant KPIs, charts, insights, and a 3-month rolling forecast.

import io
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st

# Prefer Plotly; gracefully fall back to Altair if Plotly isn't present
HAS_PLOTLY = True
try:
    import plotly.express as px
except Exception:
    HAS_PLOTLY = False
    import altair as alt

# ------------------------
# UI SETUP
# ------------------------
st.set_page_config(
    page_title="TruHome Monthly Insights",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .big-metric {font-size: 28px; font-weight: 700;}
    .subtle {color:#6b7280;}
    .insight {font-size:16px; margin-bottom:0.2rem;}
    .small-note {font-size:12px; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title("📈 TruHome Monthly Insights")
st.caption("Upload your CSV → automatic KPIs, charts, insights, and a 3-month rolling forecast. Clean & investor-friendly.")

# ------------------------
# Helpers
# ------------------------
def normalize_colname(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    x = x.strip().lower()
    x = re.sub(r"[\s\.\-_/]+", " ", x)
    x = re.sub(r"[^\w ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def month_label(dt: pd.Timestamp) -> str:
    return dt.strftime("%B - %Y")

def to_month_start(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt, errors="coerce")
    return (dt.dt.to_period("M").dt.to_timestamp()).astype("datetime64[ns]")

def detect_schema(df: pd.DataFrame) -> dict:
    """Detect aggregated monthly table vs raw leads and map key columns."""
    norm = {c: normalize_colname(c) for c in df.columns}

    store_candidates = {c for c,n in norm.items() if n in {
        "store location","store","location","walmart store","walmart location","store_name","storelocation"
    } or ("store" in n and "promoter" not in n)}

    month_text_candidates = {c for c,n in norm.items() if n in {"month year","monthyear","month","month - year"}}
    date_candidates = {c for c,n in norm.items() if n in {"collected at","collected_at","date","created at","created_at"} or "date" in n}

    total_leads_candidates = {c for c,n in norm.items() if n in {"total leads collected","total leads","leads"}}
    homeowners_total_candidates = {c for c,n in norm.items() if n in {"total homeowners","homeowners"}}
    renters_total_candidates = {c for c,n in norm.items() if n in {"total renters","renters"}}
    conversion_candidates = {c for c,n in norm.items() if "conversion" in n or n in {"homeowner conversion","homeowner conversion %","homeowner conversion percent"}}

    homeowner_flag_candidates = {c for c,n in norm.items() if n in {"homeowner","homeowner?","is homeowner","homeowner status"} or "homeowner" in n}
    renter_flag_candidates = {c for c,n in norm.items() if n in {"renter","renters?","is renter"} or "renter" in n}
    status_candidates = {c for c,n in norm.items() if n in {"status","lead status"}}

    is_aggregated = (len(total_leads_candidates) > 0) and (len(month_text_candidates | date_candidates) > 0) and (len(store_candidates) > 0)

    mapping = {
        "store": next(iter(store_candidates), None),
        "month_text": next(iter(month_text_candidates), None),
        "date": next(iter(date_candidates), None),
        "total_leads": next(iter(total_leads_candidates), None),
        "total_homeowners": next(iter(homeowners_total_candidates), None),
        "total_renters": next(iter(renters_total_candidates), None),
        "conversion": next(iter(conversion_candidates), None),
        "homeowner_flag": next(iter(homeowner_flag_candidates), None),
        "renter_flag": next(iter(renter_flag_candidates), None),
        "status": next(iter(status_candidates), None),
        "is_aggregated": bool(is_aggregated),
    }
    return mapping

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return normalized monthly store summary with:
    ['month','store','leads','homeowners','renters','conversion']
    (No disqualified — per your request)
    """
    mp = detect_schema(df)

    # Aggregated monthly file
    if mp["is_aggregated"]:
        month_col = mp["month_text"] if mp["month_text"] else mp["date"]
        temp = df.copy()

        if month_col == mp["date"]:
            temp["__month"] = month_label(to_month_start(temp[month_col]))
        else:
            temp["__month"] = temp[month_col].astype(str)

        store = mp["store"]
        leads = mp["total_leads"]
        h = mp["total_homeowners"]
        r = mp["total_renters"]
        conv = mp["conversion"]

        # Ensure numeric
        for c in [leads, h, r]:
            if c and c in temp.columns:
                temp[c] = pd.to_numeric(temp[c], errors="coerce")

        out = temp.groupby(["__month", store], dropna=False).agg(
            leads=(leads, "sum"),
            homeowners=(h, "sum") if h in temp.columns else (leads, "sum"),
            renters=(r, "sum") if r in temp.columns else (leads, "sum"),
        ).reset_index()

        # Conversion
        if conv and conv in temp.columns:
            conv_map = temp[[ "__month", store, conv ]].copy()
            conv_map[conv] = pd.to_numeric(conv_map[conv].astype(str).str.replace("%","", regex=False), errors="coerce")/100.0
            conv_agg = conv_map.groupby(["__month", store])[conv].mean().reset_index()
            out = out.merge(conv_agg, on=["__month", store], how="left")
            out.rename(columns={conv:"conversion"}, inplace=True)
        else:
            out["conversion"] = np.where(out["leads"]>0, out["homeowners"]/out["leads"], np.nan)

        out.rename(columns={"__month":"month", store:"store"}, inplace=True)
        return out

    # Raw leads
    temp = df.copy()
    store = mp["store"]
    date_col = mp["date"] or mp["month_text"]
    if date_col is None:
        raise ValueError("Could not detect a date/month column in the uploaded file.")

    if date_col == mp["month_text"]:
        temp["month"] = temp[date_col].astype(str)
    else:
        temp["month"] = month_label(to_month_start(temp[date_col]))

    def to_bool(s):
        return s.astype(str).str.strip().str.lower().isin(["yes","y","true","1"])

    homeowner = None
    renter = None

    if mp["homeowner_flag"] and mp["homeowner_flag"] in temp.columns:
        homeowner = to_bool(temp[mp["homeowner_flag"]])
    elif mp["status"] and mp["status"] in temp.columns:
        homeowner = temp[mp["status"]].astype(str).str.lower().str.contains("homeowner")

    if mp["renter_flag"] and mp["renter_flag"] in temp.columns:
        renter = to_bool(temp[mp["renter_flag"]])
    elif mp["status"] and mp["status"] in temp.columns:
        renter = temp[mp["status"]].astype(str).str.lower().str.contains("renter")

    temp["homeowner_bin"] = homeowner if homeowner is not None else False
    temp["renter_bin"] = renter if renter is not None else False

    out = temp.groupby(["month", store], dropna=False).agg(
        leads=("month", "count"),
        homeowners=("homeowner_bin", "sum"),
        renters=("renter_bin", "sum"),
    ).reset_index()
    out["conversion"] = np.where(out["leads"]>0, out["homeowners"]/out["leads"], np.nan)
    out.rename(columns={store:"store"}, inplace=True)
    return out

# --- Chart helpers (Plotly or Altair) ---
def show_bar(df, x, y, title, color=None, stacked=False, text=None):
    if df is None or df.empty:
        return
    if HAS_PLOTLY:
        fig = px.bar(df, x=x, y=y, color=color, title=title,
                     barmode="stack" if stacked else "relative", text=text)
        if text: fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="", margin=dict(l=10,r=10,t=60,b=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        enc = {"x": x, "y": y}
        if color: enc["color"] = color
        chart = alt.Chart(df).mark_bar().encode(**enc).properties(title=title)
        st.altair_chart(chart, use_container_width=True)

def show_line(df, x, y, title):
    if df is None or df.empty:
        return
    if HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, markers=True, title=title)
        fig.update_layout(xaxis_title="", yaxis_title="", margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = alt.Chart(df).mark_line(point=True).encode(x=x, y=y).properties(title=title)
        st.altair_chart(chart, use_container_width=True)

# --- Forecast using last 3 months only (rolling mean) ---
def forecast_rolling_mean(summary_df: pd.DataFrame, store: str, periods: int = 1):
    """
    Simple and stable with tiny history:
    Train on the **last 3 months** of actuals for the store.
    Predict next N months as that 3-month average.
    """
    ts = summary_df[summary_df["store"] == store].copy()
    if ts.empty:
        return None, None, "No data for selected store."

    ts["_dt"] = pd.to_datetime("01 " + ts["month"], format="%d %B - %Y", errors="coerce")
    ts = ts.sort_values("_dt").tail(3)
    if len(ts) == 0:
        return None, None, "Insufficient data."
    avg = float(ts["leads"].mean())

    last_month_dt = ts["_dt"].max()
    hist = ts.rename(columns={"leads":"leads"}).copy()
    hist = hist[["month","_dt","leads"]].rename(columns={"_dt":"date"})

    # Build forecast rows
    rows = []
    for k in range(1, periods+1):
        dt = last_month_dt + relativedelta(months=+k)
        rows.append({"date": dt, "month": month_label(dt), "leads": avg})
    fc = pd.DataFrame(rows)
    return hist, fc, None

# ------------------------
# SIDEBAR — Inputs
# ------------------------
with st.sidebar:
    st.header("Upload & Filters")
    file = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Tip: export from Google Sheets or Airtable as CSV.")

    st.divider()
    top_n = st.slider("Top N stores for charts", 3, 20, 5, step=1)

    st.checkbox("Show forecast (per selected store)", value=True, key="do_fc")
    selected_store_for_fcast = st.text_input("Forecast store (exact name)", value="")
    forecast_horizon = st.slider("Forecast horizon (months)", 1, 3, 1, step=1)  # <= 3 as requested

if not file:
    st.info("Upload a CSV to begin. The app recognizes either **raw leads** or **monthly summary** formats.")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(file)
except UnicodeDecodeError:
    df = pd.read_csv(file, encoding="latin-1")

# Prepare normalized monthly summary (no disqualified)
try:
    summary = prepare_data(df)
except Exception as e:
    st.error(f"Could not parse the file automatically.\n\n**Error:** {e}")
    st.stop()

# Clean types
for col in ["leads","homeowners","renters","conversion"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce")

# Filters
months = sorted(summary["month"].dropna().unique().tolist(),
                key=lambda x: pd.to_datetime("01 " + x, format="%d %B - %Y"))
stores = sorted(summary["store"].dropna().unique().tolist())

col_f1, col_f2 = st.columns([1,3])
with col_f1:
    month_sel = st.selectbox("Month", months, index=len(months)-1 if months else 0)
with col_f2:
    store_multi = st.multiselect("Stores (leave empty for all)", stores, default=[])

filtered = summary.copy()
if store_multi:
    filtered = filtered[filtered["store"].isin(store_multi)]
filtered_month = filtered[filtered["month"] == month_sel]

# ------------------------
# KPI Tiles (selected month)
# ------------------------
m_leads = int(filtered_month["leads"].sum())
m_home = int(filtered_month["homeowners"].sum())
m_rent = int(filtered_month["renters"].sum())
m_conv = (m_home / m_leads) if m_leads > 0 else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='big-metric'>{m_leads:,}</div><div class='subtle'>Leads</div>", unsafe_allow_html=True)
k2.markdown(f"<div class='big-metric'>{m_home:,}</div><div class='subtle'>Homeowners</div>", unsafe_allow_html=True)
k3.markdown(f"<div class='big-metric'>{m_rent:,}</div><div class='subtle'>Renters</div>", unsafe_allow_html=True)
k4.markdown(f"<div class='big-metric'>{(m_conv*100):.1f}%</div><div class='subtle'>Conversion</div>", unsafe_allow_html=True)
st.caption(f"Month shown: **{month_sel}**" + (f" · Stores: **{len(store_multi)} selected**" if store_multi else " · Stores: **All**"))

st.divider()

# ------------------------
# Insights (plain English)
# ------------------------
def safe_top(dfm, col, asc=False):
    if dfm.empty or dfm[col].notna().sum() == 0:
        return None, None
    x = dfm.sort_values(col, ascending=asc).iloc[0]
    return x["store"], x[col]

m_df = filtered_month.copy()
top_leads_store, top_leads_val = safe_top(m_df, "leads", asc=False)

m_df_nonzero = m_df[m_df["conversion"] > 0].copy()
best_conv_store, best_conv_val = safe_top(m_df_nonzero, "conversion", asc=False)
worst_conv_store, worst_conv_val = safe_top(m_df_nonzero, "conversion", asc=True)

st.subheader("Insights")
col_i1, col_i2 = st.columns(2)
with col_i1:
    if top_leads_store is not None:
        st.markdown(f"<div class='insight'>• **{top_leads_store}** led in total leads ({int(top_leads_val):,}).</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='insight'>• Overall conversion this month is <b>{(m_conv*100):.1f}%</b>.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='insight'>• Homeowners vs. Renters mix: <b>{m_home:,}</b> vs <b>{m_rent:,}</b>.</div>", unsafe_allow_html=True)
with col_i2:
    if best_conv_store is not None:
        st.markdown(f"<div class='insight'>• Best conversion: **{best_conv_store}** at <b>{best_conv_val*100:.1f}%</b>.</div>", unsafe_allow_html=True)
    if worst_conv_store is not None:
        st.markdown(f"<div class='insight'>• Lowest conversion (ex-0): **{worst_conv_store}** at <b>{worst_conv_val*100:.1f}%</b>.</div>", unsafe_allow_html=True)

st.markdown("<div class='small-note'>Tip: filter stores above to tailor the view for a region or portfolio.</div>", unsafe_allow_html=True)

st.divider()

# ------------------------
# Charts
# ------------------------
# 1) Ranking (bar)
rank_df = filtered_month.sort_values("leads", ascending=False).head(top_n)
show_bar(rank_df, x="store", y="leads", title=f"Top {len(rank_df)} Stores by Leads — {month_sel}", text="leads")

# 2) Trend over time (line) — total across selected stores
trend = filtered.gr
