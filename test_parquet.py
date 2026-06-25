import pandas as pd

df = pd.read_parquet("backtest_dataset/STOCKS/stock_data_filtered_from_10_filtered.parquet")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: f"{x:,.0f}" if abs(x) > 1000 else f"{x}")
df1 = df[
    (df["stock_float"] > 0) &
    (
        ((df["open"] < 1)  & (df["stock_float"] <= 50_000_000)) |
        ((df["open"] >= 1) & (df["stock_float"] <= 20_000_000))
    )
]

df2 = df1[['ticker','date_str','open', 'close', 'high', 'low', 'stock_float','market_cap']]

print(f"Rows: {len(df1):,}")
print(df2)  

df3 = df2[df2["open"] < 1]
print(df3)