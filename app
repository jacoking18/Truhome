# app.py — TruHome Monthly Insights (Investor View)
# Clean, simple, and robust. Upload a CSV and get instant insights + light forecasts.

import io
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional forecasting (graceful fallback if statsmodels not installed at runtime)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATS = True
except Exception:
    HAS_STATS = False


# ------------------------
# UI SETUP
# ------------------------
st.set_page_config(
    page_title="TruHome Store Insights",
    page_icon="📈",
    layout="wide",
)

# Minimal clean style
st.markdown("""
<style>
    .big-metric {font-size: 28px; font-weight: 700;}
    .subtle {color:#6b7280;}
    .insight {font-size:16px; margin-bottom:0.2rem;}
    .small-note {font-size:12px; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title("📈 TruHome Monthly Insights")
st.caption("Upload your CSV → automatic KPIs, charts, insights, and light forecasting. Clean & investor-friendly.")


# ------------------------
# Helpers
# ------------------------
def normalize_colname(x: str) -> str:
    """lowercase, remove punctuation/spaces so we can map variations robustly."""
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
    """
    Detect if the file is a raw leads table (one row per lead) or an
    aggregated monthly table (one row per store-month).
    Returns a mapping with canonical keys.
    """
    norm = {c: normalize_colname(c) for c in df.columns}

    # Candidate fields by meaning
    store_candidates = {c for c,n in norm.items() if n in {
        "store location","store","location","walmart store","walmart location","store_name","storelocation"
    } or ("store" in n and "promoter" not in n)}

    month_text_candidates = {c for c,n in norm.items() if n in {"month year","monthyear","month","month - year"}}
    date_candidates = {c for c,n in norm.items() if n in {"collected at","collected_at","date","created at","created_at"} or "date" in n}

    total_leads_candidates = {c for c,n in norm.items() if n in {"total leads collected","total leads","leads"}}
    homeowners_total_candidates = {c for c,n in norm.items() if n in {"total homeowners","homeowners"}}
    renters_total_candidates = {c for c,n in norm.items() if n in {"total renters","renters"}}
    conversion_candidates = {c for c,n in norm.items() if "conversion" in n or n in {"homeowner conversion","homeowner conversion","homeowner conversion"}}

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
    Returns a normalized monthly store summary with columns:
    ['month','store','leads','homeowners','renters','disqualified','conversion']
    """
    mp = detect_schema(df)

    # Case A: aggregated monthly file provided
    if mp["is_aggregated"]:
        month_col = mp["month_text"] if mp["month_text"] else mp["date"]
        temp = df.copy()

        if month_col == mp["date"]:
            # Convert date → month label
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
            homeowners=(h, "sum") if h in temp.columns else (leads, "sum"),  # safe default
            renters=(r, "sum") if r in temp.columns else (leads, "sum"),
        ).reset_index()

        # Disqualified (if we only have homeowners/renters, infer)
        out["disqualified"] = out["leads"] - out["homeowners"].fillna(0) - out["renters"].fillna(0)
        out["disqualified"] = out["disqualified"].clip(lower=0)

        # Conversion
        if conv and conv in temp.columns:
            # Use provided rate if it looks numeric, else compute
            # Try to coerce percent strings like "41.2%"
            conv_map = temp[[ "__month", store, conv ]].copy()
            conv_map[conv] = pd.to_numeric(conv_map[conv].astype(str).str.replace("%","", regex=False), errors="coerce")/100.0
            conv_agg = conv_map.groupby(["__month", store])[conv].mean().reset_index()
            out = out.merge(conv_agg, on=["__month", store], how="left")
            out.rename(columns={conv:"conversion"}, inplace=True)
        else:
            out["conversion"] = np.where(out["leads"]>0, out["homeowners"]/out["leads"], np.nan)

        out.rename(columns={"__month":"month", store:"store"}, inplace=True)
        return out

    # Case B: raw leads file
    # Need: date → month, store, and classification
    temp = df.copy()
    store = mp["store"]
    date_col = mp["date"] or mp["month_text"]
    if date_col is None:
        raise ValueError("Could not detect a date/month column in the uploaded file.")

    # Build month label
    if date_col == mp["month_text"]:
        temp["month"] = temp[date_col].astype(str)
    else:
        temp["month"] = month_label(to_month_start(temp[date_col]))

    # Derive classes
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
    temp["disqualified_bin"] = ~(temp["homeowner_bin"] | temp["renter_bin"])

    # Aggregate to monthly store summary
    out = temp.groupby(["month", store], dropna=False).agg(
        leads=("month", "count"),
        homeowners=("homeowner_bin", "sum"),
        renters=("renter_bin", "sum"),
        disqualified=("disqualified_bin", "sum"),
    ).reset_index()
    out["conversion"] = np.where(out["leads"]>0, out["homeowners"]/out["leads"], np.nan)
    out.rename(columns={store:"store"}, inplace=True)
    return out


def make_forecast(df_summary: pd.DataFrame, store: str, periods: int = 3):
    """
    Holt-Winters forecast for a single store's monthly leads.
    Returns (history_df, forecast_df) or (None, None) on insufficient data.
    """
    if not HAS_STATS:
        return None, None, "Forecasting library not available (statsmodels)."

    ts = df_summary[df_summary["store"] == store].copy()
    if ts.empty:
        return None, None, "No data for selected store."

    # Create a real datetime index from month labels
    ts["_dt"] = pd.to_datetime("01 " + ts["month"], format="%d %B - %Y", errors="coerce")
    ts = ts.sort_values("_dt")
    if ts["leads"].notna().sum() < 6:
        return None, None, "Need at least 6 months for a stable forecast."

    series = ts.set_index("_dt")["leads"].asfreq("MS")  # month start
    series = series.fillna(method="ffill")

    try:
        model = ExponentialSmoothing(series, trend="add", seasonal=None).fit(optimized=True)
        fcast = model.forecast(periods)
        hist = series.reset_index().rename(columns={"index":"date","leads":"leads"})
        fc = fcast.reset_index().rename(columns={"index":"date", 0:"leads"})
        fc["month"] = fc["date"].dt.strftime("%B - %Y")
        hist["month"] = hist["date"].dt.strftime("%B - %Y")
        return hist[["month","date","leads"]], fc[["month","date","leads"]], None
    except Exception as e:
        return None, None, f"Forecast failed: {e}"


# ------------------------
# SIDEBAR — Inputs
# ------------------------
with st.sidebar:
    st.header("Upload & Filters")
    file = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Tip: You can export from Google Sheets or Airtable as CSV.")

    st.divider()
    top_n = st.slider("Top N stores for charts", 3, 20, 10, step=1)
    show_forecast = st.checkbox("Show forecast (per selected store)", value=True)
    selected_store_for_fcast = st.text_input("Forecast store (exact name)", value="")
    forecast_horizon = st.slider("Forecast horizon (months)", 1, 6, 3, step=1)

if not file:
    st.info("Upload a CSV to begin. The app recognizes either **raw leads** or **monthly summary** formats.")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(file)
except UnicodeDecodeError:
    # Try latin-1 fallback
    df = pd.read_csv(file, encoding="latin-1")

# Prepare normalized monthly summary
try:
    summary = prepare_data(df)
except Exception as e:
    st.error(f"Could not parse the file automatically.\n\n**Error:** {e}")
    st.stop()

# Clean types
for col in ["leads","homeowners","renters","disqualified"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce")
summary["conversion"] = pd.to_numeric(summary["conversion"], errors="coerce")

# Global filters (auto)
months = sorted(summary["month"].dropna().unique().tolist(),
                key=lambda x: pd.to_datetime("01 " + x, format="%d %B - %Y"))
stores = sorted(summary["store"].dropna().unique().tolist())

col_f1, col_f2 = st.columns([1,3])
with col_f1:
    month_sel = st.selectbox("Month", months, index=len(months)-1 if months else 0)
with col_f2:
    store_multi = st.multiselect("Stores (leave empty for all)", stores, default=[])

# Apply filters
filtered = summary.copy()
if month_sel:
    filtered_month = filtered[filtered["month"] == month_sel]
else:
    filtered_month = filtered.copy()

if store_multi:
    filtered = filtered[filtered["store"].isin(store_multi)]
    filtered_month = filtered_month[filtered_month["store"].isin(store_multi)]

# ------------------------
# KPI Tiles (selected month)
# ------------------------
m_leads = int(filtered_month["leads"].sum())
m_home = int(filtered_month["homeowners"].sum())
m_rent = int(filtered_month["renters"].sum())
m_disq = int(filtered_month["disqualified"].sum())
m_conv = (m_home / m_leads) if m_leads > 0 else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(f"<div class='big-metric'>{m_leads:,}</div><div class='subtle'>Leads</div>", unsafe_allow_html=True)
k2.markdown(f"<div class='big-metric'>{m_home:,}</div><div class='subtle'>Homeowners</div>", unsafe_allow_html=True)
k3.markdown(f"<div class='big-metric'>{m_rent:,}</div><div class='subtle'>Renters</div>", unsafe_allow_html=True)
k4.markdown(f"<div class='big-metric'>{m_disq:,}</div><div class='subtle'>Disqualified</div>", unsafe_allow_html=True)
k5.markdown(f"<div class='big-metric'>{(m_conv*100):.1f}%</div><div class='subtle'>Conversion</div>", unsafe_allow_html=True)
st.caption(f"Month shown: **{month_sel}**" + (f" · Stores: **{len(store_multi)} selected**" if store_multi else " · Stores: **All**"))

st.divider()

# ------------------------
# Insights (plain English)
# ------------------------
# Top store by leads, highest/lowest conversion for selected month
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
if not rank_df.empty:
    fig1 = px.bar(rank_df, x="store", y="leads", title=f"Top {len(rank_df)} Stores by Leads — {month_sel}",
                  text="leads")
    fig1.update_traces(textposition="outside")
    fig1.update_layout(xaxis_title="", yaxis_title="Leads", margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig1, use_container_width=True)

# 2) Trend over time (line) — total across selected stores
trend = filtered.groupby("month", as_index=False).agg(leads=("leads","sum"),
                                                      homeowners=("homeowners","sum"))
# sort by real time
if not trend.empty:
    trend["_dt"] = pd.to_datetime("01 " + trend["month"], format="%d %B - %Y", errors="coerce")
    trend = trend.sort_values("_dt")
    fig2 = px.line(trend, x="month", y="leads", markers=True,
                   title="Leads Over Time (selected stores)")
    fig2.update_layout(xaxis_title="", yaxis_title="Leads", margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig2, use_container_width=True)

# 3) Composition (stacked) for selected month
comp = filtered_month.melt(id_vars=["store"], value_vars=["homeowners","renters","disqualified"],
                           var_name="type", value_name="count")
if not comp.empty:
    fig3 = px.bar(comp, x="store", y="count", color="type", barmode="stack",
                  title=f"Lead Composition by Store — {month_sel}")
    fig3.update_layout(xaxis_title="", yaxis_title="Count", margin=dict(l=10,r=10,t=60,b=10), legend_title="")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ------------------------
# Forecast (optional)
# ------------------------
if show_forecast:
    st.subheader("Forecast")
    if not selected_store_for_fcast:
        st.caption("Enter a store name in the sidebar to see a leads forecast.")
    else:
        hist, fc, err = make_forecast(summary, selected_store_for_fcast, periods=forecast_horizon)
        if err:
            st.warning(err)
        else:
            plot_df = pd.concat([
                hist.assign(kind="History"),
                fc.assign(kind="Forecast")
            ])
            figf = px.line(plot_df, x="month", y="leads", color="kind",
                           markers=True, title=f"Leads Forecast — {selected_store_for_fcast}")
            figf.update_layout(xaxis_title="", yaxis_title="Leads", margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(figf, use_container_width=True)

            # Quick forecast summary sentence
            last_hist = hist.iloc[-1]
            next1 = fc.iloc[0]
            change = ((next1["leads"] - last_hist["leads"]) / last_hist["leads"]) if last_hist["leads"] else np.nan
            st.markdown(
                f"<div class='insight'>• Next month forecast for <b>{selected_store_for_fcast}</b>: "
                f"<b>{int(round(next1['leads'])):,}</b> "
                f"({('+' if change>=0 else '')}{(change*100):.1f}% vs last month).</div>",
                unsafe_allow_html=True
            )

st.divider()

# ------------------------
# Downloads
# ------------------------
with st.expander("Download processed data"):
    # Summary for all months/stores
    csv_buf = io.StringIO()
    summary.to_csv(csv_buf, index=False)
    st.download_button("Download normalized summary CSV", csv_buf.getvalue(), file_name="truhome_summary.csv")

    # Month selection snapshot
    snap = filtered_month.sort_values("leads", ascending=False)
    csv_buf2 = io.StringIO()
    snap.to_csv(csv_buf2, index=False)
    st.download_button(f"Download snapshot ({month_sel})", csv_buf2.getvalue(),
                       file_name=f"snapshot_{month_sel.replace(' ','_')}.csv")

st.caption("© TruHome — Investor Preview Dashboard (MVP).")
