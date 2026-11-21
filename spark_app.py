import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from pyspark.sql import SparkSession

import dashboard
import market_views
from agent.spark_tools import make_tools
from tools.spark_analysis import load_olist

load_dotenv()
st.set_page_config(page_title="Insight | PySpark", layout="wide")
dashboard.apply_style()


@st.cache_resource
def load_data(directory):
    """Cache the item-level facts, including the order's actual and promised delivery dates."""
    spark = SparkSession.builder.master("local[*]").appName("Insight").getOrCreate()
    return load_olist(spark, Path(directory)).cache()


directory = Path(os.getenv("OLIST_DATA_DIR", str(Path(__file__).parent / "data" / "olist")))
try:
    frame = load_data(str(directory.resolve()))
except Exception as error:
    st.error(str(error))
    st.markdown("[Download the Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and extract its CSVs into the configured Olist data directory.")
    st.stop()


@st.cache_resource
def analyst(dataset_id, sales_scope, _frame):
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        timeout=60,
        model_kwargs={"parallel_tool_calls": False},
    )
    return create_agent(
        model=model, tools=make_tools(_frame),
        system_prompt=(
            "You analyze Olist e-commerce orders from 2016 to 2018. Use tools for every numeric claim. "
            "Dimensions: city, state, status, category, product_id, month. "
            "Metrics: sales, orders, items, freight, average_item_price. "
            "Sales is item price in BRL, excluding freight and before refunds. "
            "Use the delivered status for sales questions unless asked otherwise. "
            "Orders counts distinct order IDs that have item records. "
            "Use YYYY-MM-DD dates. Follow each tool's end-date convention. Ask for the year when ambiguous. "
            "Keep cities, metrics and filters from earlier messages for follow-up questions. "
            "Use sao paulo for São Paulo and rio de janeiro for Rio de Janeiro. "
            "For city totals, call analyze_sales with group_by set to [city] and all requested cities in filters.city. "
            "For a comparison with category gaps, call compare_category_sales, which returns both cities' totals and ranked gaps. "
            "Use explain_sales_change for period-over-period contributions and analyze_delivery for delivery KPIs. "
            "For questions referring to the dashboard's sales change, use its dates and filters if available. "
            "Tell the user which filters and periods support the answer. "
            "Cancellation status is available. Refunds and returns are unavailable. "
            + (f"Most recent dashboard sales-change scope: {sales_scope}" if sales_scope else "")
        ),
    )


def show_results(messages):
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        rows = payload.get("rows")
        audit = payload.get("calculation")
        if not rows and not audit:
            continue
        with st.expander(message.name.replace("_", " ").title(), expanded=True):
            if city_totals := payload.get("city_totals"):
                st.dataframe(city_totals, hide_index=True)
            if totals := payload.get("totals"):
                st.dataframe([totals], hide_index=True)
            if "change_brl" in payload:
                st.dataframe([{field: payload[field] for field in
                               ("baseline_brl", "current_brl", "change_brl", "change_pct")}], hide_index=True)
            if not rows:
                st.caption("No matching groups.")
            else:
                st.dataframe(rows, hide_index=True)
                metric = payload.get("metric", "gap")
                dimensions = [key for key in rows[0] if key != metric]
                if len(dimensions) == 1 or (metric == "gap" and "category" in dimensions):
                    x = dimensions[0] if metric != "gap" else "category"
                    chart_rows = sorted(rows, key=lambda row: row[x] or "") if x == "month" else rows
                    dashboard.plot(pd.DataFrame(chart_rows), x, metric,
                                   kind="line" if x == "month" else "bar",
                                   key=f"chat_chart_{message.tool_call_id}")
        if audit:
            dashboard.calculation_details(audit, rows, f"chat-{message.tool_call_id}")


def render_analyst():
    if not os.getenv("GROQ_API_KEY"):
        st.info("Set GROQ_API_KEY in .env to enable the analyst.")
        return

    st.caption("Try comparing São Paulo and Rio de Janeiro in October 2017 and ranking their category gaps.")
    history = st.session_state.setdefault("olist_messages", [])
    start = 0
    for index, message in enumerate(history):
        if message.type == "human":
            with st.chat_message("user"):
                st.write(message.content)
            start = index
        elif message.type == "ai" and not message.tool_calls:
            with st.chat_message("assistant"):
                st.markdown(message.text)
                show_results(history[start:index])

    if question := st.chat_input("Ask about Olist orders"):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing Olist data with PySpark..."):
                    scope = st.session_state.get("last_sales_scope")
                    response = analyst(dataset_id, json.dumps(scope) if scope else None, frame).invoke(
                        {"messages": [*history, {"role": "user", "content": question}]}
                    )
                updated = response["messages"]
                st.session_state.olist_messages = updated
                st.markdown(updated[-1].text)
                show_results(updated[len(history):])
            except Exception as error:
                st.error(f"Analysis failed: {error}")
st.session_state.setdefault("dashboard_view", "Overview")
view = st.segmented_control(
    "Dashboard view", ["Overview", "Explore", "Trends", "Compare", "Sales change", "Delivery", "Ask Insight"],
    selection_mode="single", label_visibility="collapsed", key="dashboard_view",
)
dataset_id = str(directory.resolve())
options = dashboard.filter_options(dataset_id, frame)
if not options["first"]:
    st.warning("This dataset has no item records with purchase dates.")
    st.stop()

dates, filters = dashboard.filter_panel(options)
if view == "Ask Insight":
    st.caption("Mention any dates or groups you want to analyze, or ask about the last sales-change view you opened.")
    render_analyst()
elif view == "Sales change":
    market_views.sales_change_view(dataset_id, frame, options, filters)
elif view == "Delivery" and (len(dates) == 2 or st.session_state.get("delivery_from_change")):
    market_views.delivery_view(dataset_id, frame, dates, filters)
elif len(dates) != 2:
    st.info("Choose both a start and end date.")
else:
    scoped = dashboard.scoped_frame(frame, dates, filters)
    if view == "Explore":
        dashboard.explorer(scoped, dates, filters)
    elif view == "Trends":
        dashboard.trends(scoped, options, dates, filters)
    elif view == "Compare":
        dashboard.compare(scoped, options, dates, filters)
    else:
        dashboard.overview(dataset_id, dates, filters, scoped)
