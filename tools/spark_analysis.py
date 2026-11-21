from pathlib import Path

from pyspark.sql import Window, functions as F

REQUIRED_FILES = (
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
)


def load_olist(spark, directory: Path):
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing Olist files: " + ", ".join(missing))

    def read(index, *columns, optional=()):
        source = spark.read.option("header", True).csv(
            (directory / REQUIRED_FILES[index]).resolve().as_posix()
        )
        optional_columns = [F.col(name) if name in source.columns else F.lit(None).cast("string").alias(name)
                            for name in optional]
        return source.select(*columns, *optional_columns)

    orders = read(0, "order_id", "customer_id", "order_status", "order_purchase_timestamp",
                  optional=("order_delivered_customer_date", "order_estimated_delivery_date"))
    items = read(1, "order_id", "product_id", "price", "freight_value")
    customers = read(2, "customer_id", "customer_city", "customer_state")
    products = read(3, "product_id", "product_category_name")
    categories = read(4, "product_category_name", "product_category_name_english")

    return (
        items.join(orders, "order_id")
        .join(customers, "customer_id")
        .join(products, "product_id", "left")
        .join(categories, "product_category_name", "left")
        .select(
            "order_id", "product_id",
            F.col("order_status").alias("status"),
            F.to_date(F.substring("order_purchase_timestamp", 1, 10), "yyyy-MM-dd").alias("purchase_date"),
            F.to_date(F.substring("order_delivered_customer_date", 1, 10), "yyyy-MM-dd").alias("delivered_date"),
            F.to_date(F.substring("order_estimated_delivery_date", 1, 10), "yyyy-MM-dd").alias("promised_date"),
            F.col("customer_city").alias("city"),
            F.col("customer_state").alias("state"),
            F.coalesce("product_category_name_english", "product_category_name", F.lit("unknown")).alias("category"),
            F.col("price").cast("decimal(14,2)").alias("item_sales"),
            F.col("freight_value").cast("decimal(14,2)").alias("freight"),
        )
    )


DIMENSIONS = {"city", "state", "status", "category", "product_id", "month"}
METRICS = {
    "sales": lambda: F.sum("item_sales"),
    "orders": lambda: F.countDistinct("order_id"),
    "items": lambda: F.count("*"),
    "freight": lambda: F.sum("freight"),
    "average_item_price": lambda: F.round(F.avg("item_sales"), 2),
}


def analyze(frame, metric, group_by=(), filters=None, start=None, end=None,
            limit=20, ascending=False):
    from datetime import date

    filters = filters or {}
    if metric not in METRICS or set(group_by) - DIMENSIONS or set(filters) - DIMENSIONS:
        raise ValueError("Unsupported metric or dimension.")
    if len(set(group_by)) != len(group_by) or not 1 <= limit <= 100:
        raise ValueError("Use distinct dimensions and a limit between 1 and 100.")
    start = date.fromisoformat(start) if start else None
    end = date.fromisoformat(end) if end else None
    if start and end and start >= end:
        raise ValueError("The end date must be after the start date.")

    frame = frame.withColumn("month", F.date_format("purchase_date", "yyyy-MM"))
    if start:
        frame = frame.filter(F.col("purchase_date") >= F.lit(start))
    if end:
        frame = frame.filter(F.col("purchase_date") < F.lit(end))
    for column, values in filters.items():
        if not values:
            raise ValueError(f"Specify at least one value for {column}.")
        frame = frame.filter(F.lower(F.col(column)).isin([value.strip().lower() for value in values]))

    order = F.col(metric).asc_nulls_last() if ascending else F.col(metric).desc_nulls_last()
    return frame.groupBy(*group_by).agg(METRICS[metric]().alias(metric)).orderBy(
        order, *(F.col(column).asc_nulls_last() for column in group_by)
    ).limit(limit)


def category_gap(frame, city_a, city_b, start=None, end=None, with_totals=False):
    from datetime import date

    city_a, city_b = city_a.strip().lower(), city_b.strip().lower()
    if not city_a or not city_b or city_a == city_b:
        raise ValueError("Enter two distinct cities.")

    start = date.fromisoformat(start) if start else None
    end = date.fromisoformat(end) if end else None
    if start and end and start >= end:
        raise ValueError("The end date must be after the start date.")
    delivered = frame.filter((F.col("status") == "delivered") & F.lower("city").isin(city_a, city_b))
    if start:
        delivered = delivered.filter(F.col("purchase_date") >= F.lit(start))
    if end:
        delivered = delivered.filter(F.col("purchase_date") < F.lit(end))
    city = F.lower(F.col("city"))
    gaps = (
        delivered.groupBy("category")
        .agg(
            F.coalesce(F.sum(F.when(city == city_a, F.col("item_sales"))), F.lit(0)).alias("city_a_sales"),
            F.coalesce(F.sum(F.when(city == city_b, F.col("item_sales"))), F.lit(0)).alias("city_b_sales"),
        )
        .withColumn("gap", F.abs(F.col("city_a_sales") - F.col("city_b_sales")))
    )
    if with_totals:
        all_categories = Window.partitionBy()
        gaps = gaps.select(
            "*",
            F.sum("city_a_sales").over(all_categories).alias("city_a_total"),
            F.sum("city_b_sales").over(all_categories).alias("city_b_total"),
        )
    return gaps.orderBy(F.desc("gap"), "category").limit(20)
