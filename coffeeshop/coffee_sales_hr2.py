# DataFrame: df → coffee_sales_df
# Dedup tracking: initial_rows → row_count_before_dedup
# Statistics:
# total_sales → total_coffee_sales_amount
# average_sales → average_transaction_amount
# max_sales/min_sales → maximum_transaction_amount/minimum_transaction_amount
# Analysis variables:
# product_analysis → coffee_product_sales_summary
# top_5_products_by_sales → top_five_products_by_revenue
# top_5_products_by_items_sold → top_five_products_by_quantity
# daily_sales → daily_sales_timeseries


import pandas as pd


# Load the coffee sales data 
coffee_sales_df = pd.read_csv('coffee sales dataset 2.csv')
total_coffee_sales_amount = coffee_sales_df['money'].sum()

# preview the first 5 rows
# print(coffee_sales_df.head())

# statistical summary of the dataframe
coffee_summarry = coffee_sales_df.info()
# print(coffee_summarry)
# Data Cleaning
# print("\n---Starting Data Cleaning:----")

# convert 'datetime' column from object to datetime type for time-series analysis
coffee_sales_datetime_convert = coffee_sales_df['datetime'] = pd.to_datetime(coffee_sales_df['datetime'], errors='coerce')
# print(coffee_sales_datetime_convert)
# check for missing values after conversion
missing_values_after_conversion = coffee_sales_df.isnull().sum()
# print("\n1. Missing Values in each columns after conversion:")
# print(missing_values_after_conversion)
# remove rows where 'datetime' or 'money' is missing, as they are critical for analysis
# coffee_sales_df.dropna(subset=['datetime', 'money','coffee_name'], inplace=True)

row_count_before_dedup = len(coffee_sales_df)
# print = (row_count_before_dedup)
coffee_sales_duplicate_rows = coffee_sales_df.drop_duplicates(inplace=True)
# print(f"\n2. removed {row_count_before_dedup - len(coffee_sales_df)} duplicate rows.")

# ---Descriptive Analytics--
# print("\n---performing descriptive analytics---")
total_coffee_sales_amount = coffee_sales_df['money'].sum()
average_transaction_amount = coffee_sales_df['money'].mean()
maximum_transaction_amount = coffee_sales_df['money'].max()
minimum_transaction_amount = coffee_sales_df['money'].min()
# print(f"\n1. Total Sales: ${total_coffee_sales_amount:.2f}")
# print(f"2. Average Sales: ${average_transaction_amount:.2f}")
# print(f"3. Maximum Sales: ${maximum_transaction_amount:.2f}")
# print(f"4. Minimum Sales: ${minimum_transaction_amount:.2f}")

# analysis variables: analyze product revenue
coffee_product_sales_summary = coffee_sales_df.groupby('coffee_name')['money'].agg(['sum', 'count']).reset_index()
coffee_product_sales_summary.rename(columns={'sum': 'total_sales', 'count': 'items_sold'}, inplace=True)
# print(coffee_product_sales_summary)
#find the top 5 products by Revenue
top_five_products_by_sales = coffee_product_sales_summary.sort_values(by='total_sales', ascending=False).head(5)
# print("\n4. Top 5 Products by Sales:")
# print(top_five_products_by_sales)
# find the top 5 products by items sold
# top_five_products_by_quantity = coffee_product_sales_summary.sort_values(by='items_sold', ascending=False).head(5)

# daily sales time series trend
# daily_sales_timeseries = coffee_sales_df.set_index('datetime').resample('D')['money'].sum().reset_index()
# print(daily_sales_timeseries)
# weekly sales time series trend
weekly_sales_timeseries = coffee_sales_df.set_index('datetime').resample('W')['money'].sum().reset_index()
# print("\n Weekly Sales Time Series:")
# print(weekly_sales_timeseries)
# monthly sales time series trend
monthly_sales_timeseries = coffee_sales_df.set_index('datetime').resample('M')['money'].sum().reset_index()
# print("\n Monthly Sales Time Series:")
# print(monthly_sales_timeseries)

# --- Visualizations---
# print("\n---performing visualizations---")
#set the style for the plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# a.Bar chart for top 5 products by revenue
plt.figure(figsize=(10, 6))
sns.barplot(data=top_five_products_by_sales, x='coffee_name', y='total_sales', color='brown')
plt.title('Top 5 Products by Revenue', fontsize=16)
plt.xlabel('Product', fontsize=12)
plt.ylabel('Total Sales($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_5_products_by_revenue.png')
# print("\n1. saved 'top_5_products_by_revenue.png'")
plt.show()

# b.Bar chart for top 5 products by items sold
top_five_products_by_quantity = coffee_product_sales_summary.sort_values(by='items_sold', ascending=False).head(5)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_five_products_by_quantity, x='coffee_name', y='items_sold', color='orange')
plt.title('Top 5 Products by Items Sold', fontsize=16)
plt.xlabel('Product', fontsize=12)
plt.ylabel('Items Sold', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_5_products_by_items_sold.png')
print("\n2. saved 'top_5_products_by_items_sold.png'")
plt.show()