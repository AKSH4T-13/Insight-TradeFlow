from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard
from tools.marketplace_metrics import delivery_performance, sales_change


@st.cache_data(show_spinner="Calculating sales contributions…")
def cached_change(dataset, reference, current, dimension, filters, _frame):
    return sales_change(_frame, reference, current, dimension, filters)


@st.cache_data(show_spinner="Calculating delivery performance…")
def cached_delivery(dataset, dates, dimension, filters, _frame):
    return delivery_performance(_frame, dates, dimension, filters)


def label(value):
    return "Unknown" if value is None else value.replace("_", " ").title()


def waterfall(result):
    rows = result["rows"]
    visible = rows[:8]
    names = ["Baseline total"] + [label(row["group"]) for row in visible]
    values = [result["baseline_brl"]] + [row["change_brl"] for row in visible]
    steps = ["absolute"] + ["relative"] * len(visible)
    if len(rows) > len(visible):
        names.append("Remaining groups")
        values.append(sum(row["change_brl"] for row in rows[len(visible):]))
        steps.append("relative")
    names.append("Current total")
    values.append(0)
    steps.append("total")
    figure = go.Figure(go.Waterfall(
        x=names, y=[float(value) for value in values], measure=steps,
        increasing={"marker": {"color": "#08776c"}},
        decreasing={"marker": {"color": "#b84d65"}},
        totals={"marker": {"color": "#284b69"}},
        connector={"line": {"color": "#90a6a5"}},
        yhoverformat=",.2f",
    ))
    figure.update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff",
                         yaxis_tickprefix="R$ ", yaxis_gridcolor="#e8eeea",
                         margin=dict(l=30, r=25, b=45, t=20), height=390)
    st.plotly_chart(figure, key="sales_waterfall", theme=None, width="stretch")


def contribution_picker(rows, key):
    visible = rows[:8]
    figure = go.Figure(go.Bar(
        x=[float(row["change_brl"]) for row in visible], y=[label(row["group"]) for row in visible],
        orientation="h", marker_color=["#08776c" if row["change_brl"] >= 0 else "#b84d65" for row in visible],
    ))
    figure.update_layout(template="plotly_white", clickmode="event+select", xaxis_tickprefix="R$ ",
                         yaxis_autorange="reversed", height=390, margin=dict(l=30, r=25, b=45, t=20))
    selection = st.plotly_chart(figure, key=key, theme=None, on_select="rerun", selection_mode="points")
    points = selection.selection.points
    index = points[0].get("point_index", points[0].get("point_number")) if points else None
    return visible[index]["group"] if index is not None and 0 <= index < len(visible) else None


def open_delivery(dimension, group, dates, filters):
    st.session_state["delivery_from_change"] = (dimension, group, dates, filters)
    st.session_state["dashboard_view"] = "Delivery"


def sales_change_view(dataset, frame, options, filters):
    st.subheader("Sales change")
    st.caption("Compare two purchase periods for delivered sales. Sidebar geography and category filters apply.")
    first, last = options["first"], options["last"]
    defaults = ((date(2017, 10, 1), date(2017, 10, 31)), (date(2017, 11, 1), date(2017, 11, 30)))
    midpoint = first + (last - first) // 2
    if first > defaults[0][0] or defaults[1][1] > last:
        if first == last:
            st.info("At least two purchase dates are required to compare periods.")
            return
        defaults = ((first, midpoint), (midpoint + date.resolution, last))
    columns = st.columns(3)
    baseline = columns[0].date_input("Baseline period", defaults[0], first, last)
    current = columns[1].date_input("Comparison period", defaults[1], first, last)
    dimension = columns[2].selectbox("Explain by", ["city", "category", "state"], format_func=dashboard.DIMENSIONS.get)
    if len(baseline) != 2 or len(current) != 2:
        st.info("Select both dates for each period.")
        return
    dimensions = {field: values for field, values in filters.items() if field != "status"}
    try:
        result = cached_change(dataset, baseline, current, dimension, dimensions, frame)
    except ValueError as error:
        st.warning(str(error))
        return
    if not result["rows"]:
        st.info("No delivered item records match these periods and filters.")
        return
    left, right, change = st.columns(3)
    left.metric("Baseline sales", dashboard.format_value(result["baseline_brl"], "sales"))
    right.metric("Comparison sales", dashboard.format_value(result["current_brl"], "sales"))
    pct = result["change_pct"]
    change.metric("Net change", dashboard.format_value(result["change_brl"], "sales"),
                  delta=f"{pct:+.2f}%" if pct is not None else None)
    st.session_state["last_sales_scope"] = result["calculation"]
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("**Sales movement**")
        waterfall(result)
    with right:
        st.markdown("**Select a contribution to drill down**")
        selected = contribution_picker(result["rows"], f"contributions_{dimension}_{baseline}_{current}_{dimensions}")
    st.caption("Charts show the eight largest changes by magnitude. The waterfall sums all remaining contributions into one bar.")
    st.dataframe(result["rows"], hide_index=True, width="stretch")
    dashboard.calculation_details(result["calculation"], result["rows"], "sales-change")

    drill_dimension = "city" if dimension == "state" else "category" if dimension == "city" else "city"
    groups = [row["group"] for row in result["rows"] if row["group"] is not None]
    group = st.selectbox(f"Drill into {dashboard.DIMENSIONS[dimension].lower()}", [None, *groups],
                         index=groups.index(selected) + 1 if selected in groups else 0,
                         format_func=lambda value: label(value) if value else "Select a group",
                         key=f"sales_drill_{dimension}_{selected}")
    if group is None:
        return
    drill_filters = {**dimensions, dimension: [group]}
    if dimension in {"state", "city"}:
        st.button("View delivery performance", on_click=open_delivery, args=(dimension, group, current, drill_filters))
    drill = cached_change(dataset, baseline, current, drill_dimension, drill_filters, frame)
    st.session_state["last_sales_scope"] = drill["calculation"]
    st.subheader(f"{label(group)} by {dashboard.DIMENSIONS[drill_dimension].lower()}")
    st.dataframe(drill["rows"], hide_index=True, width="stretch")
    dashboard.calculation_details(drill["calculation"], drill["rows"], "sales-drill")


def delivery_view(dataset, frame, dates, filters):
    st.subheader("Delivery performance")
    linked = st.session_state.get("delivery_from_change")
    if linked:
        dimension, group, dates, filters = linked
        st.caption(f"{label(group)} · {dates[0]:%d %b %Y} to {dates[1]:%d %b %Y} · Opened from Sales change")
        if st.button("Use sidebar scope"):
            del st.session_state["delivery_from_change"]
            st.rerun()
    dimension = st.selectbox("Break down delivery by", ["state", "city"],
                             index=1 if linked and linked[0] == "city" else 0,
                             format_func=dashboard.DIMENSIONS.get)
    result = cached_delivery(dataset, dates, dimension, filters, frame)
    totals = result["totals"]
    if not totals.get("delivered_orders"):
        st.info("No delivered orders match this scope.")
        return
    rate = totals["on_time_pct"]
    median = totals["median_delivery_days"]
    a, b, c = st.columns(3)
    a.metric("On-time delivery", f"{rate:.2f}%" if rate is not None else "—")
    b.metric("Late orders", f"{totals['late_orders']:,}" if rate is not None else "—")
    c.metric("Median delivery time", f"{median:g} days" if median is not None else "—")
    st.caption(f"{totals['orders_with_promise']:,} of {totals['delivered_orders']:,} delivered orders have both delivery dates.")
    data = pd.DataFrame(result["rows"])
    if totals["orders_with_promise"]:
        chart = data.loc[data["orders_with_promise"] > 0].head(12).copy()
        chart["group"] = chart["group"].map(label)
        dashboard.plot(chart, "on_time_pct", "group", kind="horizontal", key="delivery_rate")
    st.dataframe(data, hide_index=True, width="stretch")
    dashboard.calculation_details(result["calculation"], result["rows"], "delivery")
