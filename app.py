# app.py — TruHome Monthly Insights (Investor View)
# Upload or auto-load a default CSV → clean KPIs, insights, charts,
# single-store mode, and next-month forecasts (overall & per store).

import io
import re
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st

# ---------- SETTINGS ----------
DEFAULT_CSV_PATH = "sample/truhome_default.csv"  # put your default CSV here
APP_TITLE = "📈 TruHome Monthly Insights"
ALL_MONTHS = "— All months —"

# Prefer Plotly; gracefully fall back to Altair if unavailable
HAS_PLOTLY = True
try:
    import plotly.express as px
except Exception:
    HAS_PLOTLY = False
    import altair as alt

# ---------- PAGE ----------
st.set_page_config(page_title="TruHome Monthly Insights", page_icon="📈", layout="wide")
st.markdown("""
<style>
.big-metric {font-size: 28px; font-weight: 700;}
.subtle {color:#6b7280;}
.insight {font-size:16px; margin-bottom:0.2rem;}
.small-note {font-size:12px; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption("Upload your CSV (or use the built-in default) → automatic KPIs, insights, clean charts, and next-month forecasts. Investor-friendly and robust to messy data.")

# ---------- HELPERS ----------
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

def month_to_dt(m: str) -> pd.Timestamp:
    return pd.to_datetime("01 " + str(m), format="%d %B - %Y", errors="coerce")

@st.cache_data(show_spinner=False)
def read_csv_safely(file_like, encoding="utf-8"):
    try:
        return pd.read_csv(file_like, encoding=encoding)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_like, encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None

def detect_schema(df: pd.DataFrame) -> dict:
    norm = {c: normalize_colname(c) for c in df.columns}
    store_candidates = {c for c,n in norm.items() if n in {
        "store location","store","location","walmart store","walmart location","store_name","storelocation"
    } or ("store" in n and "promoter" not in n)}
    month_text_candidates = {c for c,n in norm.items() if n in {"month year","monthyear","month","month - year"}}
    date_candidates = {c for c,n in norm.items() if n in {"collected at","collected_at","date","created at","created_at"} or "date" in n}
    total_leads_candidates = {c for c,n in norm.items() if n in {"total leads collected","total leads","leads"}}
    homeowners_total_candidates = {c for c,n in norm.items() if n in {"total homeowners","homeowners"}}
    renters_total_candidates = {c for c,n in norm.items() if n in {"total renters","renters"}}
    conversion_candidates = {c for c,n in norm.items() if "conversion" in n or n in {
        "homeowner conversion","homeowner conversion %","homeowner conversion percent"
    }}
    homeowner_flag_candidates = {c for c,n in norm.items() if n in {"homeowner","homeowner?","is homeowner","homeowner status"} or "homeowner" in n}
    renter_flag_candidates = {c for c,n in norm.items() if n in {"renter","renters?","is renter"} or "renter" in n}
    status_candidates = {c for c,n in norm.items() if n in {"status","lead status"}}

    is_aggregated = (len(total_leads_candidates) > 0) and (len(month_text_candidates | date_candidates) > 0) and (len(store_candidates) > 0)
    return {
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

def coerce_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.replace("%","", regex=False).str.replace(",","", regex=False).str.strip(),
        errors="coerce"
    )

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize into: ['month','store','leads','homeowners','renters','conversion']."""
    mp = detect_schema(df)
    if mp["store"] is None:
        raise ValueError("Could not detect a 'Store' column.")

    if mp["is_aggregated"]:
        month_col = mp["month_text"] if mp["month_text"] else mp["date"]
        temp = df.copy()
        if month_col == mp["date"]:
            temp["__month"] = month_label(pd.to_datetime(temp[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp())
        else:
            temp["__month"] = temp[month_col].astype(str)

        store = mp["store"]; leads = mp["total_leads"]; h = mp["total_homeowners"]; r = mp["total_renters"]; conv = mp["conversion"]
        for c in [leads, h, r]:
            if c and c in temp.columns:
                temp[c] = coerce_numeric(temp[c])

        out = temp.groupby(["__month", store], dropna=False).agg(
            leads=(leads, "sum"),
            homeowners=(h, "sum") if h in temp.columns else (leads, "sum"),
            renters=(r, "sum") if r in temp.columns else (leads, "sum"),
        ).reset_index()

        if conv and conv in temp.columns:
            conv_map = temp[[ "__month", store, conv ]].copy()
            conv_map[conv] = coerce_numeric(conv_map[conv]) / 100.0
            conv_agg = conv_map.groupby(["__month", store])[conv].mean().reset_index()
            out = out.merge(conv_agg, on=["__month", store], how="left")
            out.rename(columns={conv:"conversion"}, inplace=True)
        else:
            out["conversion"] = np.where(out["leads"]>0, out["homeowners"]/out["leads"], np.nan)

        out.rename(columns={"__month":"month", store:"store"}, inplace=True)
    else:
        temp = df.copy()
        date_col = mp["date"] or mp["month_text"]
        if date_col is None:
            raise ValueError("Could not detect a date/month column in the uploaded file.")
        if date_col == mp["month_text"]:
            temp["month"] = temp[date_col].astype(str)
        else:
            dt = pd.to_datetime(temp[date_col], errors="coerce")
            temp["month"] = month_label(dt.dt.to_period("M").dt.to_timestamp())

        store = mp["store"]
        def to_bool(s): return s.astype(str).str.strip().str.lower().isin(["yes","y","true","1"])
        homeowner = renter = None
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

    out["leads"] = coerce_numeric(out["leads"])
    out["homeowners"] = coerce_numeric(out["homeowners"])
    out["renters"] = coerce_numeric(out["renters"])
    out["conversion"] = pd.to_numeric(out["conversion"], errors="coerce")
    out.loc[out["conversion"] > 1, "conversion"] = out.loc[out["conversion"] > 1, "conversion"] / 100.0
    out["_dt"] = out["month"].apply(month_to_dt)
    out = out.sort_values(["store","_dt"])
    return out

# ---- CHART HELPERS ----
def show_bar(df, x, y, title, color=None, stacked=False, text=None):
    if df is None or df.empty:
        return
    if HAS_PLOTLY:
        fig = px.bar(df, x=x, y=y, color=color, title=title,
                     barmode=("stack" if stacked else "relative"), text=text)
        if text: fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="", margin=dict(l=10,r=10,t=60,b=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        enc = {"x": x, "y": y}
        if color: enc["color"] = color
        st.altair_chart(alt.Chart(df).mark_bar().encode(**enc).properties(title=title), use_container_width=True)

def show_line(df, x, y, title, color=None, percent=False):
    if df is None or df.empty:
        return
    if HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
        if percent:
            fig.update_yaxes(tickformat=".0%")
        fig.update_layout(xaxis_title="", yaxis_title="", margin=dict(l=10,r=10,t=60,b=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        enc = {"x": x, "y": y}
        if color: enc["color"] = color
        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(**enc).properties(title=title), use_container_width=True)

# ---- FORECASTS (next month) ----
def forecast_next_month_store(summary_df: pd.DataFrame, store: str):
    ts = summary_df[summary_df["store"] == store].copy()
    if ts.empty:
        return None, None, "No data for selected store."
    ts = ts.sort_values("_dt").tail(3)
    if ts.empty:
        return None, None, "Insufficient data."
    avg = float(ts["leads"].mean())
    last_dt = ts["_dt"].max()
    next_dt = last_dt + relativedelta(months=+1)
    hist = ts[["month","_dt","leads"]].rename(columns={"_dt":"date"})
    fc = pd.DataFrame([{"date": next_dt, "month": month_label(next_dt), "leads": avg}])
    return hist, fc, None

def forecast_next_month_overall(summary_df: pd.DataFrame, subset=None):
    """
    Overall forecast for next month across stores (or a subset of stores if provided).
    Forecasts leads/homeowners/renters via 3-month rolling mean on monthly totals.
    Returns (hist_df, fc_row, None or error string)
    """
    df = summary_df.copy()
    if subset is not None:
        df = df[df["store"].isin(subset)]
    # monthly totals
    monthly = df.groupby("month", as_index=False).agg(
        leads=("leads","sum"),
        homeowners=("homeowners","sum"),
        renters=("renters","sum")
    )
    monthly["_dt"] = monthly["month"].apply(month_to_dt)
    monthly = monthly.sort_values("_dt").tail(3)
    if monthly.empty:
        return None, None, "Insufficient data."
    next_dt = monthly["_dt"].max() + relativedelta(months=+1)
    hist = monthly.rename(columns={"_dt":"date"})[["month","date","leads","homeowners","renters"]]
    fc = pd.DataFrame([{
        "date": next_dt,
        "month": month_label(next_dt),
        "leads": float(monthly["leads"].mean()),
        "homeowners": float(monthly["homeowners"].mean()),
        "renters": float(monthly["renters"].mean())
    }])
    return hist, fc, None

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Upload & Filters")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Tip: export from Google Sheets or Airtable as CSV.")

    st.divider()
    top_n = st.slider(
        "Top stores to show (month view)",
        1, 20, 5, step=1,
        help="Used only when zero or multiple stores are selected."
    )
    exclude_zero = st.checkbox(
        "Exclude 0 / blank from conversion charts & insights",
        value=True,
        help="Hides 0/blank conversion values."
    )
    st.checkbox("Show forecast (overall or per selected store)", value=True, key="do_fc")

# ---------- LOAD DATA (upload or default) ----------
df = read_csv_safely(uploaded) if uploaded else read_csv_safely(DEFAULT_CSV_PATH)
if df is None:
    st.error("Could not read a CSV. Upload a file or add a default CSV at `sample/truhome_default.csv`.")
    st.stop()

# Normalize
try:
    summary = prepare_data(df)
except Exception as e:
    st.error(f"Could not parse the file automatically.\n\n**Error:** {e}")
    st.stop()

# Dropdown values
months = sorted(summary["month"].dropna().unique().tolist(), key=lambda m: month_to_dt(m))
stores = sorted(summary["store"].dropna().unique().tolist())

# Forecast store dropdown (after data is available)
with st.sidebar:
    if st.session_state.get("do_fc", False):
        opts = ["— Select a store —"] + stores
        fc_choice = st.selectbox("Forecast store (optional)", opts, index=0, key="fc_store",
                                 help="Pick a store for a per-store forecast. Leave as 'Select a store' to see overall forecast.")
        selected_store_for_fcast = "" if fc_choice == "— Select a store —" else fc_choice
        st.caption("Forecasts show **next month** using a 3-month rolling average.")

# Main filters
c1, c2 = st.columns([1,3])
with c1:
    month_sel = st.selectbox("Month", [ALL_MONTHS] + months, index=len(months))  # latest month by default
with c2:
    store_multi = st.multiselect("Stores (leave empty for all)", stores, default=[])

# Build filtered frames
filtered = summary.copy()
if store_multi:
    filtered = filtered[filtered["store"].isin(store_multi)]

if month_sel == ALL_MONTHS:
    filtered_month = filtered.copy()
else:
    filtered_month = filtered[filtered["month"] == month_sel]

# single-store mode?
single_store_mode = (len(store_multi) == 1)
single_store_name = store_multi[0] if single_store_mode else None

# ---------- KPI TILES ----------
m_leads = int(coerce_numeric(filtered_month["leads"]).sum() or 0)
m_home  = int(coerce_numeric(filtered_month["homeowners"]).sum() or 0)
m_rent  = int(coerce_numeric(filtered_month["renters"]).sum() or 0)
m_conv  = (m_home / m_leads) if m_leads > 0 else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='big-metric'>{m_leads:,}</div><div class='subtle'>Leads</div>", unsafe_allow_html=True)
k2.markdown(f"<div class='big-metric'>{m_home:,}</div><div class='subtle'>Homeowners</div>", unsafe_allow_html=True)
k3.markdown(f"<div class='big-metric'>{m_rent:,}</div><div class='subtle'>Renters</div>", unsafe_allow_html=True)
k4.markdown(f"<div class='big-metric'>{(m_conv*100):.1f}%</div><div class='subtle'>Conversion</div>", unsafe_allow_html=True)

scope_caption = (
    ("All months" if month_sel == ALL_MONTHS else month_sel) +
    (" · Store: " + single_store_name if single_store_mode else (" · Stores: " + (str(len(store_multi)) + " selected" if store_multi else "All")))
)
st.caption(scope_caption)
st.divider()

# ---------- INSIGHTS ----------
st.subheader("Insights")

def store_insights(df_store: pd.DataFrame, exclude_zero_conv=True) -> list[str]:
    out = []
    if df_store.empty:
        return out
    d = df_store.copy().sort_values("_dt")

    def best(col, ignore_zero=False):
        x = d.copy()
        if ignore_zero:
            x = x[x[col] > 0]
        if x.empty or x[col].isna().all():
            return None
        row = x.loc[x[col].idxmax()]
        return row["month"], row[col]

    # worst conversion
    def worst_conv():
        x = d[d["conversion"] > 0] if exclude_zero_conv else d.dropna(subset=["conversion"])
        if x.empty:
            return None
        row = x.loc[x["conversion"].idxmin()]
        return row["month"], row["conversion"]

    for col, label in [("leads","Leads"), ("homeowners","Homeowners"), ("renters","Renters")]:
        res = best(col)
        if res:
            m, v = res; out.append(f"• Best <b>{label}</b> month: <b>{m}</b> ({int(v):,}).")
    res = best("conversion", ignore_zero=True)
    if res:
        m, v = res; out.append(f"• Best <b>Conversion</b> month: <b>{m}</b> at <b>{v*100:.1f}%</b>.")
    res = worst_conv()
    if res:
        m, v = res; out.append(f"• Lowest <b>Conversion</b> month: <b>{m}</b> at <b>{v*100:.1f}%</b>.")
    return out

def month_insights(df_m: pd.DataFrame, exclude_zero_conv=True) -> list[str]:
    out = []
    if df_m.empty:
        return out
    # top store by leads in this selection scope
    x = df_m.sort_values("leads", ascending=False)
    s, v = x.iloc[0]["store"], int(x.iloc[0]["leads"])
    out.append(f"• <b>{s}</b> led in total leads ({v:,}).")
    # overall conversion
    leads = int(df_m["leads"].sum()); homes = int(df_m["homeowners"].sum())
    conv = (homes/leads) if leads>0 else np.nan
    out.append(f"• Overall conversion is <b>{conv*100:.1f}%</b>.")
    out.append(f"• Homeowners vs. Renters mix: <b>{homes:,}</b> vs <b>{int(df_m['renters'].sum()):,}</b>.")
    # best/worst conversion stores
    z = df_m[df_m["conversion"] > 0] if exclude_zero_conv else df_m.dropna(subset=["conversion"])
    if not z.empty:
        best = z.sort_values("conversion", ascending=False).iloc[0]
        out.append(f"• Best conversion: <b>{best['store']}</b> at <b>{best['conversion']*100:.1f}%</b>.")
        worst = z.sort_values("conversion", ascending=True).iloc[0]
        out.append(f"• Lowest conversion (ex-0): <b>{worst['store']}</b> at <b>{worst['conversion']*100:.1f}%</b>.")
    return out

def overall_insights(df_scope: pd.DataFrame, exclude_zero_conv=True) -> list[str]:
    """Insights across ALL MONTHS in scope (no fixed month)."""
    out = []
    if df_scope.empty:
        return out
    # best month for total leads across all stores
    monthly = df_scope.groupby("month", as_index=False).agg(leads=("leads","sum"),
                                                            homeowners=("homeowners","sum"),
                                                            renters=("renters","sum"))
    monthly["_dt"] = monthly["month"].apply(month_to_dt)
    mrow = monthly.loc[monthly["leads"].idxmax()]
    out.append(f"• Best <b>Leads</b> month overall: <b>{mrow['month']}</b> ({int(mrow['leads']):,}).")
    # best conversion month (weighted)
    monthly["conv"] = np.where(monthly["leads"]>0, monthly["homeowners"]/monthly["leads"], np.nan)
    mcv = monthly[monthly["conv"]>0] if exclude_zero_conv else monthly
    if not mcv.empty:
        brow = mcv.loc[mcv["conv"].idxmax()]
        out.append(f"• Best <b>Conversion</b> month overall: <b>{brow['month']}</b> at <b>{brow['conv']*100:.1f}%</b>.")
    # top store by total leads across all months
    srow = df_scope.groupby("store", as_index=False).agg(leads=("leads","sum")).sort_values("leads", ascending=False).iloc[0]
    out.append(f"• Top store overall: <b>{srow['store']}</b> with <b>{int(srow['leads']):,}</b> leads.")
    return out

if single_store_mode:
    dstore = filtered[filtered["store"] == single_store_name]
    for s in store_insights(dstore, exclude_zero_conv=exclude_zero):
        st.markdown(f"<div class='insight'>{s}</div>", unsafe_allow_html=True)
else:
    if month_sel == ALL_MONTHS:
        for s in overall_insights(filtered, exclude_zero_conv=exclude_zero):
            st.markdown(f"<div class='insight'>{s}</div>", unsafe_allow_html=True)
    else:
        for s in month_insights(filtered_month, exclude_zero_conv=exclude_zero):
            st.markdown(f"<div class='insight'>{s}</div>", unsafe_allow_html=True)

st.markdown("<div class='small-note'>Tip: select exactly one store to switch to store-over-time mode, or choose “All months” for overall view.</div>", unsafe_allow_html=True)
st.divider()

# ---------- CHARTS ----------
if single_store_mode:
    dstore = filtered[filtered["store"] == single_store_name].copy()
    if not dstore.empty:
        melt = dstore.melt(id_vars=["month"], value_vars=["leads","homeowners","renters"],
                           var_name="metric", value_name="value").sort_values("month", key=lambda s: s.apply(month_to_dt))
        show_line(melt, x="month", y="value", color="metric",
                  title=f"{single_store_name} — Leads/Homeowners/Renters over time")
        conv_df = dstore.copy()
        if exclude_zero: conv_df = conv_df[conv_df["conversion"] > 0]
        show_line(conv_df.sort_values("_dt"), x="month", y="conversion",
                  title=f"{single_store_name} — Conversion % over time", percent=True)
else:
    if month_sel == ALL_MONTHS:
        # Top stores by total leads across all months
        totals = filtered.groupby("store", as_index=False).agg(leads=("leads","sum")).sort_values("leads", ascending=False).head(top_n)
        show_bar(totals, x="store", y="leads", title=f"Top {len(totals)} Stores by Total Leads — All months", text="leads")
        # Trend of total leads over time
        trend = filtered.groupby("month", as_index=False).agg(leads=("leads","sum")).sort_values("month", key=lambda s: s.apply(month_to_dt))
        show_line(trend, x="month", y="leads", title="Leads Over Time (All months)")
    else:
        rank_df = filtered_month.sort_values("leads", ascending=False).head(top_n)
        show_bar(rank_df, x="store", y="leads", title=f"Top {len(rank_df)} Stores by Leads — {month_sel}", text="leads")
        trend = filtered.groupby("month", as_index=False).agg(leads=("leads","sum")).sort_values("month", key=lambda s: s.apply(month_to_dt))
        show_line(trend, x="month", y="leads", title="Leads Over Time (selected stores)")
        comp = filtered_month.melt(id_vars=["store"], value_vars=["homeowners","renters"],
                                   var_name="type", value_name="count")
        show_bar(comp, x="store", y="count", color="type", stacked=True,
                 title=f"Lead Composition by Store — {month_sel}")

st.divider()

# ---------- FORECASTS (next month) ----------
if st.session_state.get("do_fc"):
    if single_store_mode or selected_store_for_fcast:
        st.subheader("Per-Store Forecast — Next Month (3-month rolling average)")
        store_for_fc = selected_store_for_fcast or single_store_name
        hist, fc, err = forecast_next_month_store(summary, store_for_fc)
        if err:
            st.warning(err)
        else:
            plot_df = pd.concat([hist.assign(kind="History"), fc.assign(kind="Forecast")])
            show_line(plot_df, x="month", y="leads", title=f"Leads Forecast — {store_for_fc}")
            last_hist = hist.iloc[-1]; next1 = fc.iloc[0]
            change = ((next1["leads"] - last_hist["leads"]) / last_hist["leads"]) if last_hist["leads"] else np.nan
            st.markdown(
                f"<div class='insight'>• Next month for <b>{store_for_fc}</b>: "
                f"<b>{int(round(next1['leads'])):,}</b> leads "
                f"({('+' if change>=0 else '')}{(change*100):.1f}% vs last month). "
                f"Method: 3-month rolling average.</div>", unsafe_allow_html=True
            )
    else:
        st.subheader("Overall Forecast — Next Month (3-month rolling average)")
        # Overall = across selected stores; if none selected, it's all stores
        subset = store_multi if store_multi else None
        hist, fc, err = forecast_next_month_overall(summary, subset=subset)
        if err:
            st.warning(err)
        else:
            # chart: total leads history + forecast
            plot_df = pd.concat([hist.assign(kind="History"), fc.assign(kind="Forecast")])
            show_line(plot_df, x="month", y="leads", title="Total Leads Forecast — Overall")
            # KPI-style sentence with H/O/R and conversion
            lh, hh, rh = hist.iloc[-1]["leads"], hist.iloc[-1]["homeowners"], hist.iloc[-1]["renters"]
            lf, hf, rf = fc.iloc[0]["leads"], fc.iloc[0]["homeowners"], fc.iloc[0]["renters"]
            conv_last = (hh/lh) if lh else np.nan
            conv_next = (hf/lf) if lf else np.nan
            st.markdown(
                f"<div class='insight'>• Next month overall forecast: "
                f"<b>{int(round(lf)):,}</b> leads, "
                f"<b>{int(round(hf)):,}</b> homeowners, "
                f"<b>{int(round(rf)):,}</b> renters. "
                f"Conversion ~ <b>{(conv_next*100):.1f}%</b> "
                f"({('+' if (conv_next - conv_last) >= 0 else '')}{((conv_next - conv_last)*100):.1f} pp vs last month}). "
                f"Method: 3-month rolling average.</div>",
                unsafe_allow_html=True
            )

st.divider()

# ---------- DOWNLOADS ----------
with st.expander("Download processed data"):
    buf = io.StringIO()
    summary[["month","store","leads","homeowners","renters","conversion"]].to_csv(buf, index=False)
    st.download_button("Download normalized summary CSV", buf.getvalue(), file_name="truhome_summary.csv")

    snap = filtered_month.sort_values("leads", ascending=False)[["month","store","leads","homeowners","renters","conversion"]]
    buf2 = io.StringIO(); snap.to_csv(buf2, index=False)
    st.download_button(
        f"Download snapshot ({'All_months' if month_sel==ALL_MONTHS else month_sel.replace(' ','_')})",
        buf2.getvalue(),
        file_name=f"snapshot_{'all_months' if month_sel==ALL_MONTHS else month_sel.replace(' ','_')}.csv"
    )

st.caption("© TruHome — Investor Preview Dashboard.")
