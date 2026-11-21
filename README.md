# Insight with PySpark

Insight is a Streamlit analytics dashboard for real marketplace orders from the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Spark joins orders, items, customers, products and category translations, then computes the dashboard results.

## Dashboard views

- **Overview** shows sales, distinct orders, items, average item price, monthly sales and top categories.
- **Explore** groups up to three dimensions, calculates multiple metrics, sorts results and limits the ranked rows.
- **Trends** groups results by week, month or quarter and can compare up to five series.
- **Compare** shows up to five groups side by side, with monthly trends and differences from the first selected group.
- **Sales change** compares two purchase periods for delivered orders. The waterfall reconciles changes by city, state or category. Select a contribution in the adjacent chart or picker to drill down.
- **Delivery** reports on-time rate, late delivered orders and median delivery days. Open it directly or follow the geography and current period from Sales change.
- **Ask Insight** uses Groq and the same PySpark calculations. Ask about the last dashboard sales-change view or specify new periods and filters.

Use the sidebar to filter the dashboard by purchase date, status, state, city or category.
Use **How this was calculated** to inspect date boundaries, filters and metric definitions, or export the supporting data and calculation details. The dashboard views run without an API key.

## Run locally

Use Java 17 or 21 and Python 3.11 or later. Set `JAVA_HOME` to your JDK installation.
If `data/olist/` is missing, download and extract the original Kaggle CSVs there, or
set `OLIST_DATA_DIR` to their directory. For Ask Insight, set `GROQ_API_KEY` in `.env`. The default
model is `openai/gpt-oss-120b`. Set `GROQ_MODEL` to select a different tool-calling model.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run spark_app.py
```

If Java 21 was unpacked under `.venv/jdk`, set it for the current PowerShell session before launching:

```powershell
$env:JAVA_HOME = (Get-ChildItem .venv/jdk -Directory | Select-Object -First 1).FullName
```

## Demo

1. In **Sales change**, compare October and November 2017. Filter City to São Paulo and Rio de Janeiro. Delivered item sales increase from R$123,984.46 to R$212,052.22.
2. Select São Paulo in the contribution chart. Review its category contributions, then open **View delivery performance** to see its November delivery metrics.
3. In **Ask Insight**, ask which category contributed most in the last sales-change view. Open **How this was calculated** to compare the date ranges and filters with the dashboard.

You can also try these comparisons in chat:

1. Compare delivered item sales for São Paulo and Rio de Janeiro in October 2017, then rank the biggest category gaps.
2. Ask for the same comparison in November 2017.
3. Ask for the monthly delivered sales trend in São Paulo for 2017.

Sales sum the item prices in Brazilian reais, excluding freight. Category gaps use the absolute
difference between city totals. Order counts cover orders with item records. The data provides
order status, including cancellations. It has no refund or return ledger.

The original `app.py` remains available for CSV uploads.
