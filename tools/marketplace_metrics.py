from datetime import date
from decimal import Decimal

from pyspark.sql import functions as F


SALES_DEFINITION = "Sum of item prices in BRL, excluding freight, for delivered orders."
DELIVERY_DEFINITION = (
    "One record per delivered order with matching items. On-time means delivered on or before the promised date. "
    "The on-time rate uses orders with both dates. Median delivery days uses nonnegative "
    "calendar days from purchase to delivery."
)
METRIC_DEFINITIONS = {
    "sales": "Sum of item prices in BRL, excluding freight.",
    "orders": "Count of distinct order IDs with matching item records.",
    "items": "Count of matching order-item rows.",
    "freight": "Sum of item freight charges in BRL.",
    "average_item_price": "Mean item price in BRL, rounded to two decimals.",
}


def period(start, end):
    start, end = (date.fromisoformat(value) if isinstance(value, str) else value for value in (start, end))
    if start is None or end is None or start > end:
        raise ValueError("Provide a start and end date in chronological order.")
    return start, end


def date_scope(frame, dates):
    return frame.filter(F.col("purchase_date").between(*dates))


def apply_filters(frame, filters):
    for column, values in filters.items():
        if values:
            frame = frame.filter(F.lower(F.col(column)).isin([value.strip().lower() for value in values]))
    return frame


def calculation(definition, filters, *, grouping, dates, date_end="inclusive", **details):
    return {
        "source": "Olist Brazilian E-Commerce Public Dataset",
        "definition": definition,
        "purchase_dates": dates,
        "end_date": date_end,
        "filters": {field: values for field, values in (filters or {}).items() if values},
        "group_by": list(grouping),
        **details,
    }


def sales_change(frame, reference, current, dimension, filters=None):
    reference, current = period(*reference), period(*current)
    if reference[1] >= current[0]:
        raise ValueError("The baseline period must end before the comparison period starts.")
    if dimension not in {"state", "city", "category"}:
        raise ValueError("Group sales change by state, city or category.")
    filters = {**(filters or {}), "status": ["delivered"]}
    frame = apply_filters(frame, filters)
    before = F.col("purchase_date").between(*reference)
    after = F.col("purchase_date").between(*current)
    sales = F.col("item_sales")
    grouped = (
        frame.filter(before | after).groupBy(dimension)
        .agg(
            F.coalesce(F.sum(F.when(before, sales)), F.lit(0)).alias("baseline_brl"),
            F.coalesce(F.sum(F.when(after, sales)), F.lit(0)).alias("current_brl"),
        )
        .withColumn("change_brl", F.col("current_brl") - F.col("baseline_brl"))
        .orderBy(F.desc(F.abs("change_brl")), dimension)
    )
    rows = [
        {"group": row[dimension], "baseline_brl": row.baseline_brl,
         "current_brl": row.current_brl, "change_brl": row.change_brl}
        for row in grouped.collect()
    ]
    baseline = sum((row["baseline_brl"] for row in rows), Decimal("0"))
    latest = sum((row["current_brl"] for row in rows), Decimal("0"))
    audit = calculation(
        SALES_DEFINITION + " Contribution = current sales minus baseline sales for each group.",
        filters, grouping=[dimension],
        dates={"baseline": [d.isoformat() for d in reference], "current": [d.isoformat() for d in current]},
        ranking="Absolute change, largest first. Contributions retain their sign.",
    )
    return {"baseline_brl": baseline, "current_brl": latest, "change_brl": latest - baseline,
            "change_pct": round((latest - baseline) * 100 / baseline, 2) if baseline else None,
            "rows": rows, "calculation": audit}


def delivery_performance(frame, dates, dimension, filters=None):
    start, end = period(*dates)
    if dimension not in {"state", "city"}:
        raise ValueError("Group delivery performance by state or city.")
    filters = {**(filters or {}), "status": ["delivered"]}
    orders = (
        date_scope(apply_filters(frame, filters), (start, end))
        .select("order_id", dimension, "purchase_date", "delivered_date", "promised_date")
        .dropDuplicates(["order_id"])
    )
    eligible = F.col("delivered_date").isNotNull() & F.col("promised_date").isNotNull()
    days = F.datediff("delivered_date", "purchase_date")
    results = (
        orders.groupingSets([(), (dimension,)], dimension)
        .agg(
            F.grouping(dimension).alias("overall"),
            F.count("*").alias("delivered_orders"),
            F.count_if(eligible).alias("orders_with_promise"),
            F.count_if(eligible & (F.col("delivered_date") <= F.col("promised_date"))).alias("on_time_orders"),
            F.count_if(eligible & (F.col("delivered_date") > F.col("promised_date"))).alias("late_orders"),
            F.round(F.median(F.when(days >= 0, days)), 1).alias("median_delivery_days"),
        )
        .withColumn("on_time_pct", F.round(F.try_divide("on_time_orders", "orders_with_promise") * 100, 2))
        .orderBy(F.desc("overall"), F.desc("delivered_orders"), dimension)
        .collect()
    )
    totals, rows = {}, []
    for row in results:
        values = row.asDict()
        overall = values.pop("overall")
        group = values.pop(dimension)
        if overall:
            totals = values
        else:
            rows.append({"group": group, **values})
    return {"totals": totals, "rows": rows, "calculation": calculation(
        DELIVERY_DEFINITION, filters, grouping=[dimension], dates=[start.isoformat(), end.isoformat()],
        ranking="Delivered orders, largest first.",
    )}
