"""Data input/loading functions for roast profiles

Provides utilities to load and validate roast CSV data.
"""
import pandas as pd
from pathlib import Path


def load_csv(path):
    """Load roast CSV and normalize column names.
    
    Args:
        path: str or Path to CSV file
        
    Returns:
        pd.DataFrame with normalized column names and computed 'seconds' if needed
        
    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if required columns are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    
    df = pd.read_csv(path)
    
    # Normalize column names (strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure 'seconds' column exists
    if 'seconds' not in df.columns:
        if 'time' in df.columns:
            try:
                df['seconds'] = pd.to_timedelta(df['time']).dt.total_seconds()
            except Exception as e:
                raise ValueError(
                    f"Could not convert 'time' column to seconds: {e}. "
                    "Ensure 'time' is in HH:MM:SS format or add a 'seconds' column."
                )
        else:
            raise ValueError(
                "CSV must have 'seconds' column or convertible 'time' column (HH:MM:SS)"
            )
    
    # Validate required temperature columns
    required = {'beans', 'air'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            f"Found: {set(df.columns)}"
        )
    
    return df


def get_ror_column(df):
    """Detect and return the RoR column name.
    
    Preference order: 'ror' > 'ror_f' > 'computed_air_ror'
    
    Args:
        df: pd.DataFrame with normalized columns
        
    Returns:
        str: column name, or None if not found
    """
    for col in ['ror', 'ror_f', 'computed_air_ror']:
        if col in df.columns:
            return col
    return None


def validate_roast_data(df):
    """Validate roast DataFrame structure and data quality.
    
    Args:
        df: pd.DataFrame from load_csv()
        
    Returns:
        dict with validation results
        
    Raises:
        ValueError: if validation fails critical checks
    """
    issues = []
    
    # Check row count
    if len(df) < 10:
        issues.append(f"Warning: only {len(df)} data points (typically want 100+)")
    
    # Check for NaN in key columns
    for col in ['seconds', 'beans', 'air']:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            issues.append(f"Warning: {nan_count} NaN values in '{col}'")
    
    # Check time ordering
    if not df['seconds'].is_monotonic_increasing:
        issues.append("Warning: 'seconds' not monotonically increasing")
    
    # Check temperature range (typical roaster: 50–230°C)
    for col in ['beans', 'air']:
        out_of_range = ((df[col] < 30) | (df[col] > 250)).sum()
        if out_of_range > 0:
            issues.append(
                f"Warning: {out_of_range} values in '{col}' outside 30–250°C range"
            )
    
    ror_col = get_ror_column(df)
    if ror_col is None:
        issues.append("Warning: no RoR column detected")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'row_count': len(df),
        'time_range_s': (df['seconds'].min(), df['seconds'].max()),
        'bean_range': (df['beans'].min(), df['beans'].max()),
        'air_range': (df['air'].min(), df['air'].max()),
        'ror_column': ror_col,
    }


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python input_data.py <path/to/roast.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print(f"Loading {csv_path}...")
    df = load_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    print("\nValidating data...")
    validation = validate_roast_data(df)
    print(f"✓ Valid: {validation['valid']}")
    for issue in validation['issues']:
        print(f"  - {issue}")
    
    print(f"\nData summary:")
    print(f"  Time: {validation['time_range_s'][0]:.1f}–{validation['time_range_s'][1]:.1f} s")
    print(f"  Bean temp: {validation['bean_range'][0]:.1f}–{validation['bean_range'][1]:.1f} °C")
    print(f"  Air temp: {validation['air_range'][0]:.1f}–{validation['air_range'][1]:.1f} °C")
    print(f"  RoR column: {validation['ror_column']}")
