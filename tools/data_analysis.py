from langchain_core.tools import tool
import pandas as pd

_current_dataset = None


def set_dataset(df):
    global _current_dataset
    _current_dataset = df.copy()


@tool
def analyze_dataset(
    filter_column: str = None,
    filter_value: str = None,
    group_by: str = None,
    metric: str = None,
    aggregation: str = "sum",
    sort_order: str = "desc",
    limit: int = 10
) -> dict:
    """
    Generic dataset analysis tool.

    Supports:
    - filtering
    - grouping
    - sum, mean, min, max, count
    - sorting
    - limiting results

    Use this tool for analytical questions about the dataset.
    """

    if _current_dataset is None:
        return {"error": "No dataset has been loaded."}

    df = _current_dataset.copy()

    try:
        # ---------------- FILTER ----------------
        if filter_column and filter_value is not None:

            if filter_column not in df.columns:
                return {
                    "error": f"Column '{filter_column}' not found.",
                    "available_columns": df.columns.tolist()
                }

            df = df[
                df[filter_column].astype(str).str.lower()
                == str(filter_value).lower()
            ]

            if df.empty:
                return {
                    "error": "No records match the given filter."
                }

        # ---------------- DIRECT METRIC ----------------
        if group_by is None:

            if metric is None:
                return {
                    "rows": len(df),
                    "columns": df.columns.tolist(),
                    "missing_values": df.isnull().sum().to_dict()
                }

            if metric not in df.columns:
                return {
                    "error": f"Metric '{metric}' not found."
                }

            values = pd.to_numeric(
                df[metric],
                errors="coerce"
            ).dropna()

            if values.empty:
                return {
                    "error": f"No numeric values found in '{metric}'."
                }

            if aggregation == "sum":
                value = values.sum()

            elif aggregation == "mean":
                value = values.mean()

            elif aggregation == "min":
                value = values.min()

            elif aggregation == "max":
                value = values.max()

            elif aggregation == "count":
                value = len(values)

            else:
                return {
                    "error": f"Unsupported aggregation: {aggregation}"
                }

            return {
                "metric": metric,
                "aggregation": aggregation,
                "value": round(float(value), 2)
            }

        # ---------------- GROUPED ANALYSIS ----------------
        if group_by not in df.columns:
            return {
                "error": f"Group column '{group_by}' not found."
            }

        if metric not in df.columns:
            return {
                "error": f"Metric '{metric}' not found."
            }

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

        df = df.dropna(subset=[metric])

        if aggregation == "sum":

            result = (
                df.groupby(group_by)[metric]
                .sum()
            )

        elif aggregation == "mean":

            result = (
                df.groupby(group_by)[metric]
                .mean()
            )

        elif aggregation == "min":

            result = (
                df.groupby(group_by)[metric]
                .min()
            )

        elif aggregation == "max":

            result = (
                df.groupby(group_by)[metric]
                .max()
            )

        elif aggregation == "count":

            result = (
                df.groupby(group_by)[metric]
                .count()
            )

        else:
            return {
                "error": f"Unsupported aggregation: {aggregation}"
            }

        result = result.sort_values(
            ascending=(sort_order == "asc")
        )

        result = result.head(limit)

        records = []

        for category, value in result.items():

            records.append({
                group_by: category,
                metric: round(float(value), 2)
            })

        return {
            "analysis": records,
            "filter": {
                "column": filter_column,
                "value": filter_value
            } if filter_column else None,
            "group_by": group_by,
            "metric": metric,
            "aggregation": aggregation,
            "sort_order": sort_order,
            "limit": limit
        }

    except Exception as e:

        return {
            "error": str(e)
        }

@tool
def compare_groups(
    column: str,
    group_a: str,
    group_b: str,
    metric: str,
    aggregation: str = "sum"
) -> dict:
    """
    Compare two groups from a dataset using a numeric metric.

    Examples:
    - Compare Bangalore vs Kolkata revenue
    - Compare Electronics vs Furniture quantity
    - Compare two categories by average price
    """

    if _current_dataset is None:
        return {"error": "No dataset has been loaded."}

    try:
        df = _current_dataset.copy()

        if column not in df.columns:
            return {
                "error": f"Column '{column}' not found."
            }

        # Revenue is a derived metric
        if metric.lower() == "revenue":

            if not {"Quantity", "Price"}.issubset(df.columns):
                return {
                    "error": "Revenue requires Quantity and Price columns."
                }

            df["Quantity"] = pd.to_numeric(
                df["Quantity"], errors="coerce"
            )

            df["Price"] = pd.to_numeric(
                df["Price"], errors="coerce"
            )

            df["Revenue"] = df["Quantity"] * df["Price"]

        if metric not in df.columns:
            return {
                "error": f"Metric '{metric}' not found."
            }

        df[metric] = pd.to_numeric(
            df[metric], errors="coerce"
        )

        df = df.dropna(subset=[metric])

        # Case-insensitive group matching
        group_a_df = df[
            df[column].astype(str).str.lower()
            == str(group_a).lower()
        ]

        group_b_df = df[
            df[column].astype(str).str.lower()
            == str(group_b).lower()
        ]

        if group_a_df.empty:
            return {
                "error": f"No records found for '{group_a}'."
            }

        if group_b_df.empty:
            return {
                "error": f"No records found for '{group_b}'."
            }

        def calculate(data):

            if aggregation == "sum":
                return data[metric].sum()

            elif aggregation == "mean":
                return data[metric].mean()

            elif aggregation == "min":
                return data[metric].min()

            elif aggregation == "max":
                return data[metric].max()

            elif aggregation == "count":
                return data[metric].count()

            else:
                raise ValueError(
                    f"Unsupported aggregation: {aggregation}"
                )

        value_a = calculate(group_a_df)
        value_b = calculate(group_b_df)

        difference = value_a - value_b

        if value_b != 0:
            percentage_difference = (
                difference / value_b
            ) * 100
        else:
            percentage_difference = None

        if value_a > value_b:
            winner = group_a
        elif value_b > value_a:
            winner = group_b
        else:
            winner = "Tie"

        return {
            "comparison": {
                group_a: round(float(value_a), 2),
                group_b: round(float(value_b), 2)
            },
            "metric": metric,
            "aggregation": aggregation,
            "difference": round(float(difference), 2),
            "percentage_difference": (
                round(float(percentage_difference), 2)
                if percentage_difference is not None
                else None
            ),
            "higher_group": winner
        }

    except Exception as e:
        return {
            "error": str(e)
        }

@tool
def analyze_trends(
    date_column: str,
    metric: str,
    aggregation: str = "sum",
    period: str = "month"
) -> dict:
    """
    Analyze a numeric metric over time.

    Supports:
    - month-wise trends
    - year-wise trends
    - daily trends
    - growth calculation
    """

    if _current_dataset is None:
        return {"error": "No dataset has been loaded."}

    try:
        df = _current_dataset.copy()

        if date_column not in df.columns:
            return {
                "error": f"Date column '{date_column}' not found."
            }

        # Revenue is derived
        if metric.lower() == "revenue":

            if not {"Quantity", "Price"}.issubset(df.columns):
                return {
                    "error": "Revenue requires Quantity and Price columns."
                }

            df["Quantity"] = pd.to_numeric(
                df["Quantity"], errors="coerce"
            )

            df["Price"] = pd.to_numeric(
                df["Price"], errors="coerce"
            )

            df["Revenue"] = df["Quantity"] * df["Price"]

        if metric not in df.columns:
            return {
                "error": f"Metric '{metric}' not found."
            }

        df[date_column] = pd.to_datetime(
            df[date_column],
            errors="coerce"
        )

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

        df = df.dropna(
            subset=[date_column, metric]
        )

        if df.empty:
            return {
                "error": "No valid date or metric values found."
            }

        # Create time period
        if period == "month":
            df["Period"] = df[date_column].dt.to_period("M")

        elif period == "year":
            df["Period"] = df[date_column].dt.to_period("Y")

        elif period == "day":
            df["Period"] = df[date_column].dt.date

        else:
            return {
                "error": "Period must be 'day', 'month', or 'year'."
            }

        # Aggregation
        if aggregation == "sum":

            trend = df.groupby("Period")[metric].sum()

        elif aggregation == "mean":

            trend = df.groupby("Period")[metric].mean()

        elif aggregation == "min":

            trend = df.groupby("Period")[metric].min()

        elif aggregation == "max":

            trend = df.groupby("Period")[metric].max()

        elif aggregation == "count":

            trend = df.groupby("Period")[metric].count()

        else:
            return {
                "error": f"Unsupported aggregation: {aggregation}"
            }

        trend = trend.sort_index()

        # Growth %
        growth = trend.pct_change() * 100

        records = []

        for i, value in trend.items():

            growth_value = growth.loc[i]

            records.append({
                "period": str(i),
                "value": round(float(value), 2),
                "growth_percent": (
                    round(float(growth_value), 2)
                    if pd.notna(growth_value)
                    else None
                )
            })

        # Highest / lowest period
        highest_period = trend.idxmax()
        lowest_period = trend.idxmin()

        return {
            "trend": records,
            "metric": metric,
            "aggregation": aggregation,
            "period": period,
            "highest_period": str(highest_period),
            "highest_value": round(
                float(trend.max()), 2
            ),
            "lowest_period": str(lowest_period),
            "lowest_value": round(
                float(trend.min()), 2
            )
        }

    except Exception as e:

        return {
            "error": str(e)
        }