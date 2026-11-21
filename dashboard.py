import json
from decimal import Decimal

import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark.sql import functions as F

from tools.spark_analysis import METRICS
from tools.marketplace_metrics import METRIC_DEFINITIONS, calculation

DIMENSIONS = {"state": "State", "city": "City", "category": "Category",
              "status": "Order status", "month": "Month", "product_id": "Product"}
MEASURES = {"sales": "Item sales", "orders": "Orders", "items": "Items",
            "freight": "Freight", "average_item_price": "Avg. item price"}
MONEY = {"sales", "freight", "average_item_price"}
COLORS = ["#08776c", "#3d62a1", "#ad601c", "#84529b", "#b84d65"]


def apply_style():
    st.html("""
    <style>
    [data-testid="stHeader"] { background: #f7f6f1; }
    [data-testid="stSidebar"] { border-right: 1px solid #dce4de; }
    .block-container { padding-top: 4.2rem; padding-bottom: 3rem; }
    .insight-hero {
        position: relative; isolation: isolate; overflow: hidden;
        background: radial-gradient(ellipse at 100% 0%, #27645d 0%, transparent 66%), #152e39;
        border-radius: 22px; padding: 32px clamp(22px, 4vw, 48px); margin-bottom: 10px;
    }
    .insight-hero::after {
        content: ""; position: absolute; z-index: -1; top: -246px; right: -70px;
        width: 380px; height: 380px; border-radius: 50%;
        border: 1px solid #e6fff035;
        box-shadow: 0 0 0 32px #e6fff018, 0 0 0 64px #e6fff011, 0 0 0 97px #e6fff00a;
    }
    .insight-hero span { color: #a9dfd0; font-weight: 700; letter-spacing: .16em; font-size: .72rem; }
    .insight-hero h1 { color: #fffdf6; margin: 5px 0; padding: 0; letter-spacing: -.045em;
                       font-size: clamp(2rem, 4vw, 3.35rem); }
    .insight-hero p { color: #d4e5e0; margin: 8px 0 0; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #dce5dd;
        border-top: 3px solid #8cbbad; border-radius: 15px; padding: 15px 18px;
        box-shadow: 0 4px 14px #172c3608; }
    [data-testid="stMetricLabel"] { color: #4d6464; }
    [data-testid="stMetricValue"] { color: #172c36; font-weight: 650;
        font-size: clamp(1.15rem, 2.1vw, 2rem); font-variant-numeric: tabular-nums; }
    [data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #dce5dd;
        border-radius: 16px; overflow: hidden; }
    </style>
    <header class="insight-hero">
        <span>INSIGHT / OLIST</span>
        <h1>Commerce analytics</h1>
        <p>Explore Brazilian marketplace sales, orders and freight from the Olist public dataset.</p>
    </header>
    """)


@st.cache_data(show_spinner="Loading filter options…")
def filter_options(dataset_id, _frame):
    fields = ("state", "city", "category", "status")
    row = _frame.agg(
        F.min("purchase_date").alias("first"),
        F.max("purchase_date").alias("last"),
        *(F.sort_array(F.collect_set(field)).alias(field) for field in fields),
    ).first()
    return row.asDict()


def filter_panel(options):
    with st.sidebar:
        st.header("Filters")
        dates = st.date_input("Purchase dates", (options["first"], options["last"]),
                              min_value=options["first"], max_value=options["last"],
                              key="purchase_dates")
        filters = {}
        for field in ("status", "state", "city", "category"):
            default = ["delivered"] if field == "status" and "delivered" in options[field] else []
            filters[field] = st.multiselect(
                DIMENSIONS[field], options[field], default=default, key=f"scope_{field}",
                format_func=lambda value: value.replace("_", " ").title(),
                placeholder="All",
            )
        st.caption("Empty filters include all values. Dates are inclusive.")
    return dates, filters


def scoped_frame(frame, dates, filters):
    first, last = dates
    frame = frame.filter(F.col("purchase_date").between(first, last))
    for field, values in filters.items():
        if values:
            frame = frame.filter(F.col(field).isin(values))
    return frame.withColumn("month", F.date_format("purchase_date", "yyyy-MM"))


def records(frame):
    return pd.DataFrame.from_records([
        {key: float(value) if isinstance(value, Decimal) else value for key, value in row.asDict().items()}
        for row in frame.collect()
    ]).convert_dtypes()


def plot(data, x, y, *, color=None, kind="bar", key):
    labels = {**DIMENSIONS, **MEASURES, "day": "Purchase date", "period": "Period",
              "gap": "Sales gap", "group": "Group", "on_time_pct": "On-time delivery"}
    metric = x if kind == "horizontal" else y
    money = metric in MONEY or metric == "gap"
    display = data.copy()
    for field in {x, y, color} & {"category", "city", "status"}:
        display[field] = display[field].str.replace("_", " ").str.title()
    args = dict(data_frame=display, x=x, y=y, color=color, labels=labels,
                color_discrete_sequence=COLORS, template="plotly_white")
    if kind == "line" and data[x].nunique() == 1:
        chart = px.bar(**args, barmode="group")
        chart.update_xaxes(type="category")
    elif kind == "line":
        chart = px.line(**args, markers=True)
        chart.update_traces(line_width=3, marker_size=5)
        chart.update_xaxes(tickformat="%b %Y" if x == "month" else "%d %b %Y")
    else:
        chart = px.bar(**args, orientation="h" if kind == "horizontal" else "v", barmode="group")
        if kind == "horizontal":
            chart.update_yaxes(autorange="reversed")
    chart.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                        font=dict(color="#354e54", family="Inter, system-ui, sans-serif"),
                        margin=dict(t=20, b=45, l=32, r=28), height=350,
                        hoverlabel=dict(bgcolor="#152e39", font_color="#ffffff"),
                        legend=dict(title_text="", orientation="h", y=1.05))
    chart.update_xaxes(gridcolor="#e8eeea", zeroline=False, automargin=True)
    chart.update_yaxes(gridcolor="#e8eeea", zeroline=False, automargin=True)
    update_axis = chart.update_xaxes if kind == "horizontal" else chart.update_yaxes
    if money:
        update_axis(tickprefix="R$ ")
    if metric == "on_time_pct":
        update_axis(ticksuffix="%", range=[0, 100])
    st.plotly_chart(chart, theme=None, width="stretch", key=key)


def calculation_details(audit, rows, key):
    with st.expander("How this was calculated"):
        st.write(audit["definition"])
        st.json(audit, expanded=True)
        left, right = st.columns(2)
        if rows is not None:
            left.download_button(
                "Supporting data (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                f"insight-{key}.csv", "text/csv", key=f"audit_rows_{key}", on_click="ignore",
            )
        right.download_button(
            "Calculation details (JSON)", json.dumps(audit, indent=2, default=str),
            f"insight-{key}-calculation.json", "application/json", key=f"audit_meta_{key}", on_click="ignore",
        )


def table(data, key):
    st.dataframe(data, hide_index=True, width="stretch",
                 column_config={field: label for field, label in (DIMENSIONS | MEASURES).items()})
    st.download_button("Download CSV", data.to_csv(index=False).encode("utf-8"),
                       file_name=f"insight-{key}.csv", mime="text/csv", key=f"download_{key}", on_click="ignore")


def format_value(value, metric):
    if pd.isna(value):
        return "—"
    if metric in MONEY:
        return f"R$ {value:,.2f}"
    return f"{value:,}"


@st.cache_data(show_spinner="Calculating KPIs and charts…")
def overview_data(dataset_id, dates, filters, _frame):
    metrics = ("sales", "orders", "items", "average_item_price")
    totals = _frame.agg(*(METRICS[metric]().alias(metric) for metric in metrics)).first().asDict()
    period = "day" if dates[0].replace(day=1) == dates[1].replace(day=1) else "month"
    trend_frame = _frame.withColumn("day", F.col("purchase_date")) if period == "day" else _frame
    trend = records(trend_frame.groupBy(period).agg(METRICS["sales"]().alias("sales")).orderBy(period))
    ranking = records(_frame.groupBy("category").agg(METRICS["sales"]().alias("sales"))
                      .orderBy(F.desc("sales"), "category").limit(8))
    return totals, trend, ranking, period


def overview(dataset_id, dates, filters, frame):
    totals, trend, ranking, period = overview_data(dataset_id, dates, filters, frame)
    if not totals["items"]:
        st.info("No item records match these filters.")
        return
    for column, (metric, value) in zip(st.columns(4), totals.items()):
        column.metric(MEASURES[metric], format_value(value, metric))
    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        st.subheader("Daily item sales" if period == "day" else "Monthly item sales")
        plot(trend, period, "sales", kind="line", key="overview_trend")
    with right:
        st.subheader("Top categories")
        plot(ranking, "sales", "category", kind="horizontal", key="overview_categories")
    st.caption("Sales sum item prices in BRL before refunds, excluding freight. Order counts cover distinct orders with item records.")
    evidence = [{"section": "KPI totals", **totals}]
    evidence.extend({"section": "Sales trend", **row} for row in trend.to_dict("records"))
    evidence.extend({"section": "Top categories", **row} for row in ranking.to_dict("records"))
    result_details(evidence, "overview", dates, filters, [period, "category"], list(totals), category_limit=8)


def result_details(rows, key, dates, filters, groups, metrics, **details):
    audit = calculation(" ".join(METRIC_DEFINITIONS[m] for m in metrics), filters,
                        grouping=groups, dates=[value.isoformat() for value in dates], **details)
    calculation_details(audit, rows, key)


def explorer(frame, dates, filters):
    a, b = st.columns(2)
    groups = a.multiselect("Group by", list(DIMENSIONS), default=["category"],
                          format_func=DIMENSIONS.get, max_selections=3)
    metrics = b.multiselect("Aggregations", list(MEASURES), default=["sales", "orders"],
                           format_func=MEASURES.get, max_selections=5)
    if not metrics:
        st.info("Select at least one aggregation.")
        return
    a, b, c = st.columns(3)
    sort_by = a.selectbox("Sort by", metrics + groups, format_func=(DIMENSIONS | MEASURES).get)
    ascending = b.selectbox("Sort order", ["Descending", "Ascending"]) == "Ascending"
    limit = c.slider("Top N / row limit", 1, 100, 10)
    if not st.button("Run analysis", type="primary"):
        return
    sort = F.col(sort_by).asc_nulls_last() if ascending else F.col(sort_by).desc_nulls_last()
    result = records(frame.groupBy(*groups).agg(*(METRICS[m]().alias(m) for m in metrics))
                     .orderBy(sort, *(F.asc_nulls_last(g) for g in groups if g != sort_by)).limit(limit))
    if result.empty or result.iloc[0].isna().all():
        st.info("No matching results.")
        return
    if len(groups) == 1:
        plot(result, metrics[0], groups[0], kind="horizontal", key="explorer_plot")
    table(result, "explore")
    result_details(result, "explore", dates, filters, groups, metrics, sort_by=sort_by,
                   ascending=ascending, row_limit=limit)


def trends(frame, options, dates, filters):
    a, b, c = st.columns(3)
    metric = a.selectbox("Measure", list(MEASURES), format_func=MEASURES.get, key="trend_measure")
    cadence = b.selectbox("Time interval", ["Week", "Month", "Quarter"], index=1)
    split = c.selectbox("Break down by", [None, "state", "city", "category", "status"],
                        format_func=lambda item: DIMENSIONS[item] if item else "Overall")
    values = []
    if split:
        values = st.multiselect("Series", options[split], max_selections=5, key=f"trend_series_{split}")
        if not values:
            st.info("Choose one to five series to compare over time.")
            return
    if not st.button("Run trend", type="primary"):
        return
    frame = frame.withColumn("period", F.to_date(F.date_trunc(cadence.lower(), "purchase_date")))
    if split:
        frame = frame.filter(F.col(split).isin(values))
    fields = ["period", split] if split else ["period"]
    result = records(frame.groupBy(*fields).agg(METRICS[metric]().alias(metric)).orderBy(*fields))
    if result.empty:
        st.info("No matching results.")
        return
    plot(result, "period", metric, color=split, kind="line", key="trend_plot")
    table(result, "trends")
    result_details(result, "trends", dates, filters, fields, [metric], interval=cadence,
                   selected_series=values)


def compare(frame, options, dates, filters):
    a, b = st.columns(2)
    group = a.selectbox("Compare by", ["city", "state", "category", "status"], format_func=DIMENSIONS.get)
    metric = b.selectbox("Measure", list(MEASURES), format_func=MEASURES.get, key="compare_measure")
    values = st.multiselect("Groups to compare", options[group], max_selections=5,
                            key=f"compare_groups_{group}",
                            default=["sao paulo", "rio de janeiro"] if group == "city" and
                            {"sao paulo", "rio de janeiro"} <= set(options[group]) else [])
    if len(values) < 2:
        st.info("Select at least two groups. The first group is the baseline.")
        return
    st.caption(f"Baseline: {values[0]}")
    if not st.button("Run comparison", type="primary"):
        return
    subset = frame.filter(F.col(group).isin(values))
    totals = records(subset.groupBy(group).agg(METRICS[metric]().alias(metric)))
    if totals.empty:
        st.info("No selected groups match these filters.")
        return
    totals = totals.set_index(group).reindex(values).reset_index()
    baseline = totals[metric].iloc[0]
    if pd.notna(baseline):
        totals["Difference vs baseline"] = totals[metric] - baseline
        if baseline:
            totals["Change vs baseline (%)"] = ((totals[metric] / baseline - 1) * 100).round(2)
    plot(totals, group, metric, key="comparison_totals")
    monthly = records(subset.groupBy("month", group).agg(METRICS[metric]().alias(metric)).orderBy("month", group))
    plot(monthly, "month", metric, color=group, kind="line", key="comparison_trends")
    table(totals, "compare")
    evidence = pd.concat([totals.assign(section="Group totals"), monthly.assign(section="Monthly trend")],
                         ignore_index=True)
    result_details(evidence, "compare", dates, filters, [group, "month"], [metric],
                   selected_groups=values, baseline=values[0])
