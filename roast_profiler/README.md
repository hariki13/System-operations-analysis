roast_profiler

Simple roast profile analysis tools for coffee roasting data.

Installation

```bash
pip install -r roast_profiler/requirements.txt
```

Modules

### input_data.py

Utilities for loading and validating roast CSV data. Key functions:

- `load_csv(path)` — Loads CSV, normalizes columns, converts `time` → `seconds`
- `get_ror_column(df)` — Detects RoR column (`ror` > `ror_f` > `computed_air_ror`)
- `validate_roast_data(df)` — Validates structure, data ranges, and reports issues

**Usage:**
```bash
python roast_profiler/input_data.py path/to/roast.csv
```

**In code:**
```python
from roast_profiler.input_data import load_csv, validate_roast_data

df = load_csv('roast.csv')
validation = validate_roast_data(df)
if validation['valid']:
    print(f"Loaded {validation['row_count']} rows")
```

### analyze_roast.py

Full roast analyzer: loads data, computes stats, smooths traces, detects peaks, saves plots.

**Usage:**
```bash
python roast_profiler/analyze_roast.py path/to/roast.csv
```

**Options:**
```bash
python roast_profiler/analyze_roast.py roast.csv \
  --outdir ./my_outputs \
  --smooth 7 \
  --ror-threshold 5.0
```

**Outputs:**
- `temperature_profile.png` — Bean & air temps (raw + smoothed)
- `ror_profile.png` — RoR trace with peak detection
- `summary.txt` — Roast stats (times, temps, peaks)

Features

- Handles common roaster CSV formats (multiple RoR column names)
- Auto-normalizes column names and converts time → seconds
- Validates data quality and reports issues
- Smooths temperature and RoR traces (configurable window)
- Detects RoR peaks and marks thresholds
- Generates publication-ready plots

Example Roast CSV Columns (required)

- `seconds` or `time` (HH:MM:SS format)
- `beans` (bean temperature in °C)
- `air` (air temperature in °C)
- `ror` or `ror_f` or `computed_air_ror` (rate of rise)
