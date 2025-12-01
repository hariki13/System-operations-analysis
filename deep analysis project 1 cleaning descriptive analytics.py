import os
import sys
import shutil
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("coffee sales dataset.csv")

# Optional: path where cleaned data and plots should be copied
# Set by environment variable `SAVE_TO_PATH` (example: /mnt/d/myfolder)
SAVE_TO_PATH = os.environ.get('SAVE_TO_PATH')
# Default export folder (will be used if SAVE_TO_PATH is not set)
DEFAULT_EXPORT_DIR = Path('exports')
# Whether to create a ZIP archive of the exports (can be controlled with env var ZIP_EXPORTS=0)
ZIP_EXPORTS = os.environ.get('ZIP_EXPORTS', '1') != '0'

# ===== DATA CLEANING =====
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Convert date/datetime columns to proper datetime dtype for time-based analysis
for _col in ["date", "datetime"]:
    if _col in df.columns:
        try:
            df[_col] = pd.to_datetime(df[_col])
        except Exception as e:
            print(f"Warning: could not convert column '{_col}' to datetime: {e}")

# Prepare numeric subset for numeric-only statistics and analysis
numeric_df = df.select_dtypes(include=[np.number])

# Track generated files so we can move/zip them together
generated_files = []

# Save cleaned dataset locally (workspace root) and record it
cleaned_local = Path('coffee_sales_cleaned.csv')
df.to_csv(cleaned_local, index=False)
generated_files.append(cleaned_local)
print(f"Saved cleaned dataset to {cleaned_local}")

# Determine export directory (env var overrides default)
EXPORT_DIR = Path(SAVE_TO_PATH) if SAVE_TO_PATH else DEFAULT_EXPORT_DIR

# If user explicitly requested an external SAVE_TO_PATH, require confirmation
export_allowed = False
if SAVE_TO_PATH:
    if sys.stdin.isatty():
        resp = input(f"Confirm saving outputs to external path '{EXPORT_DIR}'? [y/N]: ").strip().lower()
        export_allowed = resp in ('y', 'yes')
        if not export_allowed:
            print('Skipping export to external path.')
    else:
        # Non-interactive: require explicit AUTO_CONFIRM_SAVE=1 to proceed
        if os.environ.get('AUTO_CONFIRM_SAVE') == '1':
            export_allowed = True
        else:
            print(f"Non-interactive session: set AUTO_CONFIRM_SAVE=1 to allow saving to '{EXPORT_DIR}'")
else:
    # default export directory inside workspace is allowed without confirmation
    export_allowed = True

# ===== DESCRIPTIVE ANALYTICS =====
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe())
print("\nSkewness:\n", numeric_df.skew())
print("\nKurtosis:\n", numeric_df.kurtosis())

# ===== ADVANCED ANALYSIS =====

# 1. Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix - Coffee Sales')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
generated_files.append(Path('correlation_matrix.png'))
plt.show()

# 2. Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, col in enumerate(numeric_df.columns[:4]):
    ax = axes[idx // 2, idx % 2]
    ax.hist(numeric_df[col], bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('distributions.png', dpi=300)
generated_files.append(Path('distributions.png'))
plt.show()

# 3. Outlier Detection (IQR Method)
def detect_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()

print("\n=== OUTLIERS (IQR Method) ===")
for col in numeric_df.columns:
    outlier_count = detect_outliers(numeric_df[col])
    print(f"{col}: {outlier_count} outliers")

# 4. Statistical Testing (if categorical columns exist)
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\n=== CATEGORICAL ANALYSIS ===")
    for col in categorical_cols:
        print(f"\n{col}:\n", df[col].value_counts())

# 6. Time-series example: daily/weekly/monthly total `money` (if `date` exists and is datetime)
if 'date' in df.columns and np.issubdtype(df['date'].dtype, np.datetime64):
    # Use a time-indexed copy for resampling
    df_ts = df.set_index('date')
    daily = df_ts['money'].resample('D').sum()
    weekly = df_ts['money'].resample('W').sum()
    monthly = df_ts['money'].resample('M').sum()

    plt.figure(figsize=(12, 6))
    daily.plot()
    plt.title('Daily Total Money')
    plt.xlabel('Date')
    plt.ylabel('Total Money')
    plt.tight_layout()
    plt.savefig('daily_money_timeseries.png', dpi=300)
    generated_files.append(Path('daily_money_timeseries.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    weekly.plot(marker='o')
    plt.title('Weekly Total Money')
    plt.xlabel('Week')
    plt.ylabel('Total Money')
    plt.tight_layout()
    plt.savefig('weekly_money_timeseries.png', dpi=300)
    generated_files.append(Path('weekly_money_timeseries.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    monthly.plot(marker='o')
    plt.title('Monthly Total Money')
    plt.xlabel('Month')
    plt.ylabel('Total Money')
    plt.tight_layout()
    plt.savefig('monthly_money_timeseries.png', dpi=300)
    generated_files.append(Path('monthly_money_timeseries.png'))
    plt.show()

    # Optionally create an interactive HTML if Plotly is installed
    try:
        import plotly.express as px
        fig = px.line(monthly.reset_index(), x='date', y='money', title='Monthly Total Money')
        out_html = Path('monthly_money_timeseries.html')
        fig.write_html(str(out_html))
        generated_files.append(out_html)
        print('Saved interactive monthly plot to monthly_money_timeseries.html')
    except Exception:
        print('Plotly not available — skipped interactive HTML output.')

# 5. Advanced Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
numeric_df.boxplot(ax=axes[0])
axes[0].set_title('Box Plot - Numerical Features')

# Pair plot for key relationships
if len(numeric_df.columns) >= 2:
    axes[1].scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], alpha=0.6)
    axes[1].set_xlabel(numeric_df.columns[0])
    axes[1].set_ylabel(numeric_df.columns[1])
    axes[1].set_title('Scatter Plot - Key Relationship')

plt.tight_layout()
plt.savefig('advanced_analysis.png', dpi=300)
generated_files.append(Path('advanced_analysis.png'))
plt.show()

# After generating files, copy them to EXPORT_DIR (if allowed) and optionally zip
if export_allowed:
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        for f in generated_files:
            if f.exists():
                dest = EXPORT_DIR / f.name
                shutil.copy(f, dest)
        print(f"Copied {len(generated_files)} files to {EXPORT_DIR}")

        if ZIP_EXPORTS:
            zip_path = EXPORT_DIR.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for f in generated_files:
                    if f.exists():
                        zf.write(f, arcname=f.name)
            print(f"Created zip archive {zip_path}")
    except Exception as e:
        print(f"Warning: export/copy failed: {e}")

print("\n✓ Analysis Complete!")


