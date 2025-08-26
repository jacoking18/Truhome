# app.py — TruHome Monthly Insights (Investor View)
# Upload or auto-load a default CSV → clean KPIs, insights, charts,
# single-store mode, and a next-month forecast (3-month rolling avg).

import io
import re
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st

# ---------- SETTINGS ----------
# If no file is uploaded, we will try to load this CSV from your repo.
# Put your preferred default file at this path.
DEFAULT_CSV_PATH = "sample/truhome_default.csv"
APP_TITLE = "📈 TruHome Monthly Insights"

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
st.caption("Upload your CSV (or use the built-in default) → automatic KPIs, insights, clean charts, and a next-month forecast. Investor-friendly and robust to messy data.")

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
    """Read csv from upload or path; try latin-1 fallback; return DataFrame or None."""
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
    """Robust numeric coercion: strips %, commas and spaces."""
    return pd.to_numeric(
        s.astype(str).str.replace("%","", regex=False).str.replace(",","", regex=False).str.strip(),
        errors="coerce"
    )

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize into: ['month','store','leads','homeowners','renters','conversion'].
    conversion is a fraction in [0,1] or NaN (never >1).
    """
    mp = detect_schema(df)
    if mp["store"] is None:
        raise ValueError("Could not detect a 'Store' column.")

    # AGGREGATED MONTHLY
    if mp["is_aggregated"]:
        month_col = mp["month_text"] if mp["month_text"] else mp["date"]
        temp = df.copy()
        if month_col == mp["date"]:
            temp["__month"] = temp[month_col]
            temp["__month"] = month_label(pd.to_datetime(temp["__month"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        else:
            temp["__month"] = temp[month_col].astype(str)

        store = mp["store"]; leads = mp["total_leads"]; h = mp["total_homeowners"]; r = mp["total_renters"]; conv = mp["conversion"]
        # clean numbers
        for c in [leads, h, r]:
            if c and c in temp.columns:
                temp[c] = coerce_numeric(temp[c])

        out = temp.groupby(["__month", store], dropna=False).agg(
            leads=(leads, "sum"),
            homeowners=(h, "sum") if h in temp.columns else (leads, "sum"),
            renters=(r, "sum") if r in temp.columns else (leads, "sum"),
        ).reset_index()

        # conversion
        if conv and conv in temp.columns:
            conv_map = temp[[ "__month", store, conv ]].copy()
            conv_map[conv] = coerce_numeric(conv_map[conv]) / 100.0
            conv_agg = conv_map.groupby(["__month", store])[conv].mean().reset_index()
            out = out.merge(conv_agg, on=["__month", store], how="left")
            out.rename(columns={conv:"conversion"}, inplace=True)
        else:
            out["conversion"] = np.where(out["leads"]>0, out["homeowners"]/out["leads"], np.nan)

        out.rename(columns={"__month":"month", store:"store"}, inplace=True)

    # RAW LEADS
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
        def to_bool(s):
            return s.astype(str).str.strip().str.lower().isin(["yes","y","true","1"])
        homeowner = None; renter = None
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

    # Final clean-ups
    out["leads"] = coerce_numeric(out["leads"])
    out["homeowners"] = coerce_numeric(out["homeowners"])
    out["renters"] = coerce_numeric(out["renters"])
    out["conversion"] = pd.to_numeric(out["conversion"], errors="coerce")
    # If conversion accidentally came as >1 (e.g., 41 not 0.41), scale down
    out.loc[out["conversion"] > 1, "conversion"] = out.loc[out["conversion"] > 1, "conversion"] / 100.0
    # month order helper
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

# ---- FORECAST: next month using 3-month rolling average ----
def forecast_next_month(summary_df: pd.DataFrame, store: str):
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
        help="Keeps charts readable and avoids misleading conversion values."
    )
    st.checkbox("Show forecast (per selected store)", value=True, key="do_fc")

# ---------- LOAD DATA (upload or default) ----------
df = None
if uploaded:
    df = read_csv_safely(uploaded)
else:
    df = read_csv_safely(DEFAULT_CSV_PATH)

if df is None:
    st.error("Could not read a CSV. Upload a file or add a default CSV at `sample/truhome_default.csv` in the repo.")
    st.stop()

# Normalize
try:
    summary = prepare_data(df)
except Exception as e:
    st.error(f"Could not parse the file automatically.\n\n**Error:** {e}")
    st.stop()

# Dropdown values
months = summary["month"].dropna().unique().tolist()
months = sorted(months, key=lambda m: month_to_dt(m))
stores = sorted(summary["store"].dropna().unique().tolist())

# Forecast store dropdown (after data is available)
with st.sidebar:
    if st.session_state.get("do_fc", False):
        opts = ["— Select a store —"] + stores
        fc_choice = st.selectbox("Forecast store", opts, index=0, key="fc_store",
                                 help="Choose a store to see next-month forecast.")
        selected_store_for_fcast = "" if fc_choice == "— Select a store —" else fc_choice
        st.caption("Forecast shows **next month only** using a 3-month rolling average.")
    else:
        selected_store_for_fcast = ""

# Main filters
c1, c2 = st.columns([1,3])
with c1:
    month_sel = st.selectbox("Month", months, index=len(months)-1 if months else 0)
with c2:
    store_multi = st.multiselect("Stores (leave empty for all)", stores, default=[])

# Filtered frames
filtered = summary.copy()
if store_multi:
    filtered = filtered[filtered["store"].isin(store_multi)]
filtered_month = filtered[filtered["month"] == month_sel]

# Are we in single-store mode?
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
st.caption(
    f"Month shown: **{month_sel}**"
    + (f" · Store: **{single_store_name}**" if single_store_mode else (f" · Stores: **{len(store_multi)} selected**" if store_multi else " · Stores: **All**"))
)
st.divider()

# ---------- INSIGHTS ----------
st.subheader("Insights")

def store_insights(df_store: pd.DataFrame, exclude_zero_conv=True) -> list[str]:
    out = []
    if df_store.empty:
        return out
    d = df_store.copy().sort_values("_dt")
    # best months
    def best(col, ignore_zero=False):
        x = d.copy()
        if ignore_zero:
            x = x[x[col] > 0]
        if x.empty or x[col].isna().all():
            return None
        row = x.loc[x[col].idxmax()]
        return row["month"], row[col]
    # worst conversion (excl 0)
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
    # top leads store
    x = df_m.sort_values("leads", ascending=False)
    s, v = x.iloc[0]["store"], int(x.iloc[0]["leads"])
    out.append(f"• <b>{s}</b> led in total leads ({v:,}).")
    # overall conversion
    leads = int(df_m["leads"].sum())
    homes = int(df_m["homeowners"].sum())
    conv = (homes/leads) if leads>0 else np.nan
    out.append(f"• Overall conversion this month is <b>{conv*100:.1f}%</b>.")
    out.append(f"• Homeowners vs. Renters mix: <b>{homes:,}</b> vs <b>{int(df_m['renters'].sum()):,}</b>.")
    # best/worst conversion stores
    z = df_m[df_m["conversion"] > 0] if exclude_zero_conv else df_m.dropna(subset=["conversion"])
    if not z.empty:
        best = z.sort_values("conversion", ascending=False).iloc[0]
        out.append(f"• Best conversion: <b>{best['store']}</b> at <b>{best['conversion']*100:.1f}%</b>.")
        worst = z.sort_values("conversion", ascending=True).iloc[0]
        out.append(f"• Lowest conversion (ex-0): <b>{worst['store']}</b> at <b>{worst['conversion']*100:.1f}%</b>.")
    return out

if single_store_mode:
    dstore = filtered[filtered["store"] == single_store_name]
    for s in store_insights(dstore, exclude_zero_conv=exclude_zero):
        st.markdown(f"<div class='insight'>{s}</div>", unsafe_allow_html=True)
else:
    for s in month_insights(filtered_month, exclude_zero_conv=exclude_zero):
        st.markdown(f"<div class='insight'>{s}</div>", unsafe_allow_html=True)

st.markdown("<div class='small-note'>Tip: select exactly one store above to switch to store-over-time mode.</div>", unsafe_allow_html=True)
st.divider()

# ---------- CHARTS ----------
if single_store_mode:
    dstore = filtered[filtered["store"] == single_store_name].copy()
    if not dstore.empty:
        # Multi-line: leads/home/renters
        melt = dstore.melt(id_vars=["month"], value_vars=["leads","homeowners","renters"],
                           var_name="metric", value_name="value").sort_values("month", key=lambda s: s.apply(month_to_dt))
        show_line(melt, x="month", y="value", color="metric",
                  title=f"{single_store_name} — Leads/Homeowners/Renters over time")
        # Conversion trend (percent axis). Optionally drop zeros.
        conv_df = dstore.copy()
        if exclude_zero: conv_df = conv_df[conv_df["conversion"] > 0]
        show_line(conv_df.sort_values("_dt"), x="month", y="conversion",
                  title=f"{single_store_name} — Conversion % over time", percent=True)
else:
    rank_df = filtered_month.sort_values("leads", ascending=False).head(top_n)
    show_bar(rank_df, x="store", y="leads", title=f"Top {len(rank_df)} Stores by Leads — {month_sel}", text="leads")
    trend = filtered.groupby("month", as_index=False).agg(leads=("leads","sum"))
    trend = trend.sort_values("month", key=lambda s: s.apply(month_to_dt))
    show_line(trend, x="month", y="leads", title="Leads Over Time (selected stores)")
    comp = filtered_month.melt(id_vars=["store"], value_vars=["homeowners","renters"],
                               var_name="type", value_name="count")
    show_bar(comp, x="store", y="count", color="type", stacked=True,
             title=f"Lead Composition by Store — {month_sel}")

st.divider()

# ---------- FORECAST (next month only) ----------
if st.session_state.get("do_fc"):
    st.subheader("Forecast — Next Month (3-month rolling average)")
    # If in single-store mode and no explicit forecast store chosen, use that store
    if (st.session_state.get("fc_store") in (None, "— Select a store —")) and single_store_mode:
        selected_store_for_fcast = single_store_name

    if not selected_store_for_fcast:
        st.caption("Select a store in the sidebar to see its forecast.")
    else:
        hist, fc, err = forecast_next_month(summary, selected_store_for_fcast)
        if err:
            st.warning(err)
        else:
            plot_df = pd.concat([hist.assign(kind="History"), fc.assign(kind="Forecast")])
            show_line(plot_df, x="month", y="leads", title=f"Leads Forecast — {selected_store_for_fcast}")
            last_hist = hist.iloc[-1]
            next1 = fc.iloc[0]
            change = ((next1["leads"] - last_hist["leads"]) / last_hist["leads"]) if last_hist["leads"] else np.nan
            st.markdown(
                f"<div class='insight'>• Next month forecast for <b>{selected_store_for_fcast}</b>: "
                f"<b>{int(round(next1['leads'])):,}</b> "
                f"({('+' if change>=0 else '')}{(change*100):.1f}% vs last month). "
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
    st.download_button(f"Download snapshot ({month_sel})", buf2.getvalue(), file_name=f"snapshot_{month_sel.replace(' ','_')}.csv")

st.caption("© TruHome — Investor Preview Dashboard.")
