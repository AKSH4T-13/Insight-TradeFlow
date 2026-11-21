from decimal import Decimal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict

from tools.spark_analysis import analyze, category_gap
from tools.marketplace_metrics import METRIC_DEFINITIONS, SALES_DEFINITION, calculation, delivery_performance, sales_change


class SalesFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: list[str] | None = None
    state: list[str] | None = None
    status: list[str] | None = None
    category: list[str] | None = None
    product_id: list[str] | None = None
    month: list[str] | None = None


def serialize_rows(rows):
    return [serialize_values(row.asDict()) for row in rows]


def serialize_values(values):
    return {key: float(value) if isinstance(value, Decimal) else value for key, value in values.items()}


def make_tools(frame):
    @tool
    def analyze_sales(
        metric: str,
        group_by: list[str] | None = None,
        filters: SalesFilters | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 20,
        ascending: bool = False,
    ) -> dict:
        """Aggregate Olist data. Dates are ISO, inclusive start and exclusive end."""
        active_filters = filters.model_dump(exclude_none=True) if filters else None
        try:
            rows = analyze(frame, metric, group_by or [], active_filters, start, end, limit, ascending).collect()
        except ValueError as error:
            return {"error": str(error)}
        return {
            "metric": metric,
            "filters": active_filters,
            "start": start,
            "end": end,
            "rows": serialize_rows(rows),
            "calculation": calculation(METRIC_DEFINITIONS[metric], active_filters,
                                       grouping=group_by or [], dates=[start, end], date_end="exclusive",
                                       limit=limit, ranking="Ascending" if ascending else "Descending"),
        }

    @tool
    def compare_category_sales(
        city_a: str,
        city_b: str,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        """Get total delivered item sales for two cities and rank their category sales gaps."""
        try:
            rows = serialize_rows(category_gap(frame, city_a, city_b, start, end, with_totals=True).collect())
        except ValueError as error:
            return {"error": str(error)}
        city_totals = []
        if rows:
            city_totals = [
                {"city": city_a, "sales": rows[0]["city_a_total"]},
                {"city": city_b, "sales": rows[0]["city_b_total"]},
            ]
            for row in rows:
                del row["city_a_total"], row["city_b_total"]
        return {
            "city_a": city_a, "city_b": city_b, "start": start, "end": end,
            "city_totals": city_totals, "rows": rows,
            "calculation": calculation(SALES_DEFINITION + " Category gap is the absolute difference between city sales.",
                                       {"city": [city_a, city_b], "status": ["delivered"]},
                                       grouping=["category"], dates=[start, end], date_end="exclusive",
                                       limit=20, ranking="Largest absolute category gap first"),
        }

    @tool
    def explain_sales_change(
        baseline_start: str, baseline_end: str, current_start: str, current_end: str,
        dimension: str = "city", filters: SalesFilters | None = None, limit: int = 20,
    ) -> dict:
        """Explain delivered item sales change between two periods (ISO dates, inclusive ends).

        Group by city, state or category. Filter by these fields to drill into a contribution.
        """
        filters = filters.model_dump(exclude_none=True) if filters else {}
        if set(filters) - {"state", "city", "category"} or not 1 <= limit <= 50:
            return {"error": "Use state, city or category filters and a limit from 1 to 50."}
        try:
            result = sales_change(frame, (baseline_start, baseline_end), (current_start, current_end), dimension, filters)
        except ValueError as error:
            return {"error": str(error)}
        result["calculation"].update(returned_group_limit=limit, total_groups=len(result["rows"]))
        result["rows"] = [serialize_values(row) for row in result["rows"][:limit]]
        return serialize_values(result)

    @tool
    def analyze_delivery(start: str, end: str, dimension: str = "state",
                         filters: SalesFilters | None = None, limit: int = 20) -> dict:
        """Measure on-time delivery, late orders and median delivery days (ISO dates, inclusive ends).

        Group by state or city. Counts use distinct delivered orders.
        """
        filters = filters.model_dump(exclude_none=True) if filters else {}
        if set(filters) - {"state", "city", "category"} or not 1 <= limit <= 50:
            return {"error": "Use state, city or category filters and a limit from 1 to 50."}
        try:
            result = delivery_performance(frame, (start, end), dimension, filters)
        except ValueError as error:
            return {"error": str(error)}
        result["calculation"].update(returned_group_limit=limit, total_groups=len(result["rows"]))
        result["rows"] = [serialize_values(row) for row in result["rows"][:limit]]
        result["totals"] = serialize_values(result["totals"])
        return result

    return [analyze_sales, compare_category_sales, explain_sales_change, analyze_delivery]

