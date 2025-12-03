import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

print('='*70)
print('ADVANCED: TIME SERIES, CUSTOMER BEHAVIOR & ANOMALY DETECTION')
print('='*70)

# =============================
# LOAD AND PREPARE DATA
# =============================
df = pd.read_excel('/workspaces/s/coffee_sales_data_cleaned.xlsx')

# Convert datetime columns
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = df['datetime'].dt.hour
df['day_of_month'] = df['date'].dt.day
df['week'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

print(f"\nDataset: {df.shape[0]} transactions across {df['date'].nunique()} days")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# =============================
# 1. TIME SERIES ANALYSIS
# =============================
print('\n' + '='*70)
print('1. TIME SERIES ANALYSIS')
print('='*70)

# Daily time series
daily_ts = df.groupby('date').agg({
    'money': ['sum', 'count', 'mean'],
    'cash_type': lambda x: (x == 'card').sum() / len(x) * 100
}).round(2)
daily_ts.columns = ['revenue', 'transactions', 'avg_transaction', 'card_percentage']

print("\n--- Daily Time Series Summary ---")
print(daily_ts.describe())

# Calculate rolling statistics (7-day window)
daily_ts['revenue_ma7'] = daily_ts['revenue'].rolling(window=7, min_periods=1).mean()
daily_ts['revenue_std7'] = daily_ts['revenue'].rolling(window=7, min_periods=1).std()
daily_ts['transactions_ma7'] = daily_ts['transactions'].rolling(window=7, min_periods=1).mean()

# Trend analysis
daily_ts['revenue_trend'] = daily_ts['revenue'] - daily_ts['revenue_ma7']
print("\n--- Trend Analysis ---")
print(f"Average daily growth: ${daily_ts['revenue'].diff().mean():.2f}")
print(f"Revenue volatility (std): ${daily_ts['revenue'].std():.2f}")
print(f"Best day revenue: ${daily_ts['revenue'].max():.2f} on {daily_ts['revenue'].idxmax().date()}")
print(f"Worst day revenue: ${daily_ts['revenue'].min():.2f} on {daily_ts['revenue'].idxmin().date()}")

# Seasonality - Day of week patterns
print("\n--- Weekly Seasonality ---")
weekly_pattern = df.groupby('day_of_week').agg({
    'money': ['sum', 'count', 'mean']
}).round(2)
weekly_pattern.columns = ['total_revenue', 'transactions', 'avg_transaction']
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pattern = weekly_pattern.reindex([d for d in day_order if d in weekly_pattern.index])
print(weekly_pattern)

# Hourly patterns
print("\n--- Hourly Patterns ---")
hourly_pattern = df.groupby('hour').agg({
    'money': ['sum', 'count', 'mean']
}).round(2)
hourly_pattern.columns = ['revenue', 'transactions', 'avg_transaction']
print(hourly_pattern)

# =============================
# 2. CUSTOMER BEHAVIOR ANALYSIS
# =============================
print('\n' + '='*70)
print('2. CUSTOMER BEHAVIOR ANALYSIS')
print('='*70)

# Product preference analysis
print("\n--- Product Preferences ---")
product_behavior = df.groupby('coffee_name').agg({
    'money': ['count', 'sum', 'mean'],
    'cash_type': lambda x: (x == 'card').sum() / len(x) * 100
}).round(2)
product_behavior.columns = ['purchase_count', 'total_revenue', 'avg_price', 'card_pref_%']
product_behavior['popularity_score'] = (
    product_behavior['purchase_count'] / product_behavior['purchase_count'].sum() * 100
).round(2)
product_behavior = product_behavior.sort_values('purchase_count', ascending=False)
print(product_behavior.head(10))

# Payment behavior patterns
print("\n--- Payment Method Behavior ---")
payment_by_hour = pd.crosstab(df['hour'], df['cash_type'], normalize='index') * 100
print(payment_by_hour.round(2))

# Price sensitivity analysis
print("\n--- Price Sensitivity by Time ---")
price_by_hour = df.groupby('hour')['money'].agg(['mean', 'std', 'min', 'max']).round(2)
price_by_hour['coefficient_variation'] = (price_by_hour['std'] / price_by_hour['mean'] * 100).round(2)
print(price_by_hour)

# Weekend vs Weekday behavior
print("\n--- Weekend vs Weekday Comparison ---")
weekend_comparison = df.groupby('is_weekend').agg({
    'money': ['count', 'sum', 'mean'],
    'cash_type': lambda x: (x == 'card').sum() / len(x) * 100
}).round(2)
weekend_comparison.columns = ['transactions', 'revenue', 'avg_transaction', 'card_%']
weekend_comparison.index = ['Weekday', 'Weekend']
print(weekend_comparison)

# Customer segmentation by spending
df['spending_category'] = pd.cut(df['money'], 
                                  bins=[0, 20, 25, 30, 100],
                                  labels=['Budget', 'Standard', 'Premium', 'Luxury'])
print("\n--- Customer Spending Segments ---")
spending_segments = df.groupby('spending_category').agg({
    'money': ['count', 'sum', 'mean']
}).round(2)
spending_segments.columns = ['customers', 'total_spent', 'avg_spent']
spending_segments['percentage'] = (spending_segments['customers'] / len(df) * 100).round(2)
print(spending_segments)

# =============================
# 3. ANOMALY DETECTION
# =============================
print('\n' + '='*70)
print('3. ANOMALY DETECTION')
print('='*70)

# Method 1: Statistical Z-Score (Revenue Anomalies)
print("\n--- Method 1: Statistical Z-Score Detection ---")
daily_ts['z_score'] = np.abs(stats.zscore(daily_ts['revenue']))
daily_ts['is_anomaly_zscore'] = daily_ts['z_score'] > 2.5

anomaly_days_zscore = daily_ts[daily_ts['is_anomaly_zscore']]
print(f"Detected {len(anomaly_days_zscore)} anomalous days using Z-score method")
if len(anomaly_days_zscore) > 0:
    print("\nAnomalous days:")
    print(anomaly_days_zscore[['revenue', 'transactions', 'z_score']])

# Method 2: IQR Method (Outlier Detection)
print("\n--- Method 2: IQR (Interquartile Range) Method ---")
Q1 = daily_ts['revenue'].quantile(0.25)
Q3 = daily_ts['revenue'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

daily_ts['is_anomaly_iqr'] = (daily_ts['revenue'] < lower_bound) | (daily_ts['revenue'] > upper_bound)
anomaly_days_iqr = daily_ts[daily_ts['is_anomaly_iqr']]
print(f"Detected {len(anomaly_days_iqr)} outlier days using IQR method")
print(f"Normal range: ${lower_bound:.2f} - ${upper_bound:.2f}")
if len(anomaly_days_iqr) > 0:
    print("\nOutlier days:")
    print(anomaly_days_iqr[['revenue', 'transactions']])

# Method 3: Isolation Forest (Transaction-level anomalies)
print("\n--- Method 3: Isolation Forest (ML-based) ---")
# Prepare features for anomaly detection
features_for_anomaly = df[['money', 'hour']].copy()
features_for_anomaly['is_weekend'] = df['is_weekend']
features_for_anomaly['is_card'] = (df['cash_type'] == 'card').astype(int)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_anomaly)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = iso_forest.fit_predict(features_scaled)
df['is_anomaly_ml'] = df['anomaly_score'] == -1

anomalies_ml = df[df['is_anomaly_ml']]
print(f"Detected {len(anomalies_ml)} anomalous transactions using Isolation Forest")
print(f"Anomaly rate: {len(anomalies_ml)/len(df)*100:.2f}%")

if len(anomalies_ml) > 0:
    print("\nSample anomalous transactions:")
    print(anomalies_ml[['date', 'hour', 'coffee_name', 'money', 'cash_type']].head(10))
    print("\nAnomaly characteristics:")
    print(anomalies_ml[['money', 'hour']].describe())

# Customer behavior anomalies
print("\n--- Unusual Purchase Patterns ---")
# Products purchased at unusual times
product_hour = df.groupby(['coffee_name', 'hour']).size().reset_index(name='count')
product_hour_pivot = product_hour.pivot(index='coffee_name', columns='hour', values='count').fillna(0)

# Find products with concentrated sales in specific hours (high variance)
product_variance = product_hour_pivot.var(axis=1).sort_values(ascending=False)
print("\nProducts with most concentrated purchase times:")
print(product_variance.head(5))

# =============================
# 4. ADVANCED VISUALIZATIONS
# =============================
print('\n' + '='*70)
print('4. GENERATING ADVANCED VISUALIZATIONS')
print('='*70)

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 14))

# Plot 1: Daily Revenue Time Series with Moving Average
plt.subplot(3, 3, 1)
plt.plot(daily_ts.index, daily_ts['revenue'], marker='o', linewidth=1, alpha=0.7, label='Daily Revenue')
plt.plot(daily_ts.index, daily_ts['revenue_ma7'], linewidth=2, color='red', label='7-day MA')
plt.fill_between(daily_ts.index, 
                 daily_ts['revenue_ma7'] - daily_ts['revenue_std7'],
                 daily_ts['revenue_ma7'] + daily_ts['revenue_std7'],
                 alpha=0.2, color='red')
if len(anomaly_days_zscore) > 0:
    plt.scatter(anomaly_days_zscore.index, anomaly_days_zscore['revenue'], 
                color='red', s=100, marker='X', label='Anomalies', zorder=5)
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.title('Daily Revenue Trend with Anomaly Detection')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 2: Weekly Seasonality Pattern
plt.subplot(3, 3, 2)
plt.bar(weekly_pattern.index, weekly_pattern['total_revenue'], color='steelblue')
plt.xlabel('Day of Week')
plt.ylabel('Total Revenue ($)')
plt.title('Weekly Revenue Pattern')
plt.xticks(rotation=45)

# Plot 3: Hourly Heatmap
plt.subplot(3, 3, 3)
hourly_revenue = df.pivot_table(values='money', index='hour', columns='day_of_week', aggfunc='sum', fill_value=0)
hourly_revenue = hourly_revenue[[d for d in day_order if d in hourly_revenue.columns]]
sns.heatmap(hourly_revenue, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Revenue ($)'})
plt.title('Revenue Heatmap: Hour × Day')
plt.xlabel('Day of Week')
plt.ylabel('Hour of Day')

# Plot 4: Customer Spending Distribution
plt.subplot(3, 3, 4)
plt.hist(df['money'], bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.axvline(df['money'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["money"].mean():.2f}')
plt.axvline(df['money'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${df["money"].median():.2f}')
# Mark anomalies
if len(anomalies_ml) > 0:
    for val in anomalies_ml['money'].unique()[:5]:  # Show first 5 unique anomaly values
        plt.axvline(val, color='green', linestyle=':', alpha=0.5)
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Frequency')
plt.title('Transaction Amount Distribution')
plt.legend()

# Plot 5: Product Popularity Over Time
plt.subplot(3, 3, 5)
top5_products = df['coffee_name'].value_counts().head(5).index
for product in top5_products:
    product_daily = df[df['coffee_name'] == product].groupby('date').size()
    plt.plot(product_daily.index, product_daily.values, marker='o', label=product, linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.title('Top 5 Products: Sales Trend')
plt.legend(fontsize=8)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 6: Payment Method by Hour
plt.subplot(3, 3, 6)
payment_by_hour.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'])
plt.xlabel('Hour of Day')
plt.ylabel('Percentage (%)')
plt.title('Payment Method Distribution by Hour')
plt.legend(title='Payment Type')
plt.xticks(rotation=0)

# Plot 7: Box Plot - Revenue by Day of Week
plt.subplot(3, 3, 7)
day_order_present = [d for d in day_order if d in df['day_of_week'].values]
sns.boxplot(data=df, x='day_of_week', y='money', order=day_order_present, palette='Set2')
plt.xlabel('Day of Week')
plt.ylabel('Transaction Amount ($)')
plt.title('Price Distribution by Day')
plt.xticks(rotation=45)

# Plot 8: Anomaly Detection Scatter
plt.subplot(3, 3, 8)
plt.scatter(df[~df['is_anomaly_ml']]['hour'], 
           df[~df['is_anomaly_ml']]['money'],
           alpha=0.5, s=30, label='Normal', color='blue')
plt.scatter(df[df['is_anomaly_ml']]['hour'], 
           df[df['is_anomaly_ml']]['money'],
           alpha=0.8, s=60, label='Anomaly', color='red', marker='X')
plt.xlabel('Hour of Day')
plt.ylabel('Transaction Amount ($)')
plt.title('Anomaly Detection: Hour vs Amount')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 9: Cumulative Revenue Growth
plt.subplot(3, 3, 9)
daily_ts['cumulative_revenue'] = daily_ts['revenue'].cumsum()
plt.plot(daily_ts.index, daily_ts['cumulative_revenue'], linewidth=2, color='green')
plt.fill_between(daily_ts.index, 0, daily_ts['cumulative_revenue'], alpha=0.3, color='green')
plt.xlabel('Date')
plt.ylabel('Cumulative Revenue ($)')
plt.title('Cumulative Revenue Growth')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_analysis_visualization.png', dpi=300, bbox_inches='tight')
print('\n✅ Visualization saved as: advanced_analysis_visualization.png')
plt.show()

# =============================
# 5. KEY INSIGHTS & RECOMMENDATIONS
# =============================
print('\n' + '='*70)
print('5. KEY INSIGHTS & RECOMMENDATIONS')
print('='*70)

print("\n📊 TIME SERIES INSIGHTS:")
print(f"   • Revenue trend: {'Growing' if daily_ts['revenue'].iloc[-7:].mean() > daily_ts['revenue'].iloc[:7].mean() else 'Declining'}")
print(f"   • Peak day: {weekly_pattern['total_revenue'].idxmax()}")
print(f"   • Peak hour: {hourly_pattern['revenue'].idxmax()}:00")
print(f"   • Weekend impact: {weekend_comparison.loc['Weekend', 'revenue'] - weekend_comparison.loc['Weekday', 'revenue']:.2f} more revenue")

print("\n👥 CUSTOMER BEHAVIOR:")
print(f"   • Most popular product: {product_behavior.index[0]}")
print(f"   • Card usage: {payment_stats.loc['card', 'percentage']:.1f}%")
print(f"   • Premium customers: {spending_segments.loc['Premium', 'percentage']:.1f}%")

print("\n⚠️ ANOMALIES DETECTED:")
print(f"   • Statistical outliers: {len(anomaly_days_zscore)} days")
print(f"   • ML-detected anomalies: {len(anomalies_ml)} transactions ({len(anomalies_ml)/len(df)*100:.1f}%)")
print(f"   • Most anomalous products: {anomalies_ml['coffee_name'].value_counts().head(3).to_dict() if len(anomalies_ml) > 0 else 'None'}")

print("\n💡 RECOMMENDATIONS:")
print(f"   1. Focus marketing on {weekly_pattern['total_revenue'].idxmin()} (lowest revenue day)")
print(f"   2. Staff optimization: peak hours are {hourly_pattern.nlargest(3, 'transactions').index.tolist()}")
print(f"   3. Investigate anomalous transactions for potential fraud or data errors")
print(f"   4. Promote premium products during {price_by_hour['mean'].idxmax()}:00 (highest spending hour)")
print(f"   5. Weekend promotions for card users (higher card usage: {payment_by_hour.loc[payment_by_hour.index >= 12, 'card'].mean():.1f}%)")

print('\n' + '='*70)
print('Advanced Analysis Complete!')
print('='*70)
