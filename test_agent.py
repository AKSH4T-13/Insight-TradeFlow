import pandas as pd

from tools.data_analysis import set_dataset, compare_groups, analyze_trends

df = pd.read_csv("data/sales.csv")
set_dataset(df)

result = analyze_trends.invoke({
    "date_column": "Date",
    "metric": "Revenue",
    "aggregation": "sum",
    "period": "month"
})

print("\n===== COMPARISON RESULT =====\n")
print(result)