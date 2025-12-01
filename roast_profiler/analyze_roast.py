#!/usr/bin/env python3
"""Roast profile analyzer

Usage:
    python analyze_roast.py path/to/roast.csv

Produces summary and saves plots to `outputs/`.
"""
import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

from input_data import load_csv, get_ror_column, validate_roast_data

sns.set(style="whitegrid")


def smooth(series, window=5):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def summarize(df):
    """Compute summary statistics from roast DataFrame."""
    total_time = df['seconds'].iloc[-1] - df['seconds'].iloc[0]
    start_bean = df['beans'].iloc[0]
    end_bean = df['beans'].iloc[-1]
    max_bean = df['beans'].max()
    max_bean_time = df.loc[df['beans'].idxmax(), 'seconds']
    
    ror_col = get_ror_column(df)
    if ror_col is not None:
        max_ror = df[ror_col].max()
        max_ror_time = df.loc[df[ror_col].idxmax(), 'seconds']
    else:
        max_ror = None
        max_ror_time = None
    
    return {
        'total_time_s': total_time,
        'start_bean': start_bean,
        'end_bean': end_bean,
        'max_bean': max_bean,
        'max_bean_time_s': max_bean_time,
        'ror_col': ror_col,
        'max_ror': max_ror,
        'max_ror_time_s': max_ror_time,
    }


def plot_profile(df, outdir, smooth_window=5, ror_peak_threshold=None):
    """Generate and save temperature and RoR plots."""
    os.makedirs(outdir, exist_ok=True)
    t = df['seconds']

    # Smooth traces
    beans_s = smooth(df['beans'], smooth_window)
    air_s = smooth(df['air'], smooth_window) if 'air' in df.columns else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, df['beans'], color='tab:red', alpha=0.25, label='Bean (raw)')
    ax.plot(t, beans_s, color='tab:red', label='Bean (smoothed)')
    if air_s is not None:
        ax.plot(t, df['air'], color='tab:blue', alpha=0.25, label='Air (raw)')
        ax.plot(t, air_s, color='tab:blue', label='Air (smoothed)')
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Roast Temperature Profile')
    ax.legend()
    fig.tight_layout()
    fpath = os.path.join(outdir, 'temperature_profile.png')
    fig.savefig(fpath, dpi=150)
    plt.close(fig)

    # RoR plot
    ror_col = get_ror_column(df)
    if ror_col is None:
        print('No ROR column found; skipping RoR plot')
        return [fpath]

    ror_s = smooth(df[ror_col], smooth_window)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, df[ror_col], color='tab:green', alpha=0.25, label=f'{ror_col} (raw)')
    ax2.plot(t, ror_s, color='tab:green', label=f'{ror_col} (smoothed)')
    ax2.set_xlabel('Seconds')
    ax2.set_ylabel('RoR (°C/min ?)')
    ax2.set_title('Rate of Rise (RoR)')

    # detect peaks
    try:
        peaks, props = find_peaks(ror_s.fillna(0).values, height=0)
        if len(peaks):
            ax2.scatter(t.iloc[peaks], ror_s.iloc[peaks], color='orange', s=40, label='Peaks')
    except Exception:
        peaks = []

    # optional threshold vertical lines
    if ror_peak_threshold is not None:
        over = df[ror_col] > ror_peak_threshold
        if over.any():
            first = df.loc[over.idxmax(), 'seconds']
            ax2.axvline(first, color='orange', linestyle='--', label=f'Cross {ror_peak_threshold}')

    ax2.legend()
    fig2.tight_layout()
    f2 = os.path.join(outdir, 'ror_profile.png')
    fig2.savefig(f2, dpi=150)
    plt.close(fig2)

    return [fpath, f2]


def run(path, outdir=None, smooth_window=5, ror_peak_threshold=None):
    path = Path(path)
    if outdir is None:
        outdir = Path(path.parent) / 'roast_outputs'
    else:
        outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(path)
    if 'seconds' not in df.columns:
        print('ERROR: no `seconds` column found and `time` conversion failed')
        return 1

    summary = summarize(df)
    # write summary to file
    summary_path = outdir / 'summary.txt'
    with open(summary_path, 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print('Summary:')
    for k, v in summary.items():
        print(f" - {k}: {v}")

    plot_paths = plot_profile(df, outdir, smooth_window=smooth_window, ror_peak_threshold=ror_peak_threshold)
    print('Saved plots:')
    for p in plot_paths:
        print(' -', p)
    print('Saved summary:', summary_path)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Analyze roast profile CSV')
    p.add_argument('csv', help='Path to roast CSV file')
    p.add_argument('--outdir', help='Output directory', default=None)
    p.add_argument('--smooth', type=int, help='Smoothing window (seconds)', default=5)
    p.add_argument('--ror-threshold', type=float, help='ROR threshold for annotation', default=None)
    args = p.parse_args()
    sys.exit(run(args.csv, args.outdir, args.smooth, args.ror_threshold))
