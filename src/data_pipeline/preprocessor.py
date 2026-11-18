"""
Module: preprocessor
Purpose: Preprocess Azure VM and Alibaba cluster telemetry into continuous
         30-day (720-hour) benchmark workload traces for the 4 core archetypes:
         Flat, Seasonal, Growing, and Bursty.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "workload_traces"

HORIZON_HOURS = 720  # 30 days = 720 hours (504h train + 48h val + 168h test)


def calibrate_and_generate_azure_workloads(sample_rows: int = 200000) -> dict:
    """
    Extracts empirical CPU distributions from Azure VM telemetry and generates
    continuous 30-day (720-hour) workload traces for the 4 research archetypes.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "azure_vm_cpu_readings.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw Azure dataset at {raw_path}. Run azure_loader.py first.")

    logger.info(f"Calibrating on {sample_rows:,} real Azure VM readings...")
    df = pd.read_csv(raw_path, nrows=sample_rows)

    # Real empirical parameters from Azure datacenter
    empirical_mean_cpu = float(df["avg_cpu"].mean())
    empirical_std_cpu = float(df["avg_cpu"].std())
    fleet_size = 500  # Fleet of 500 VMs
    vcpu_per_vm = 2   # Standard 2-vCPU VMs (m5.large equivalent)

    base_fleet_demand = fleet_size * vcpu_per_vm * (empirical_mean_cpu / 100.0)
    std_fleet_demand = np.sqrt(fleet_size) * vcpu_per_vm * (empirical_std_cpu / 100.0)

    logger.info(f"Azure Empirical Base Demand: {base_fleet_demand:.1f} vCPUs (std: {std_fleet_demand:.1f})")

    hours = np.arange(HORIZON_HOURS)
    np.random.seed(42)

    # 1. Seasonal / Diurnal: Strong 24-hour cycles (business hour peak, night trough)
    diurnal = 0.40 * base_fleet_demand * np.sin(2 * np.pi * (hours - 8) / 24)
    weekly = 0.15 * base_fleet_demand * np.cos(2 * np.pi * hours / 168)
    noise_seasonal = np.random.normal(0, std_fleet_demand * 0.4, size=HORIZON_HOURS)
    seasonal_series = np.maximum(base_fleet_demand + diurnal + weekly + noise_seasonal, 10.0)

    # 2. Flat: Steady-state baseline workload with low variance (CV < 0.10)
    noise_flat = np.random.normal(0, base_fleet_demand * 0.05, size=HORIZON_HOURS)
    flat_series = base_fleet_demand * 0.95 + noise_flat

    # 3. Growing: Sustained expansion (+80% growth over 30 days)
    growth_trend = np.linspace(0, base_fleet_demand * 0.80, HORIZON_HOURS)
    noise_growing = np.random.normal(0, std_fleet_demand * 0.5, size=HORIZON_HOURS)
    growing_series = base_fleet_demand * 0.70 + growth_trend + noise_growing

    # 4. Bursty: Intermittent unpredictable surges (flash spikes)
    burst_mask = np.random.rand(HORIZON_HOURS) > 0.95
    burst_spikes = burst_mask * np.random.uniform(base_fleet_demand * 0.8, base_fleet_demand * 2.5, size=HORIZON_HOURS)
    noise_bursty = np.random.normal(0, base_fleet_demand * 0.08, size=HORIZON_HOURS)
    bursty_series = (base_fleet_demand * 0.60) + noise_bursty + burst_spikes

    start_timestamp = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [start_timestamp + pd.Timedelta(hours=int(h)) for h in hours]

    profiles = {
        "azure_seasonal": seasonal_series,
        "azure_flat": flat_series,
        "azure_growing": growing_series,
        "azure_bursty": bursty_series,
    }

    saved_files = []
    for name, series in profiles.items():
        out_df = pd.DataFrame({
            "timestamp": timestamps,
            "demand_vcpu": np.round(series, 2),
            "workload_type": name.replace("azure_", "")
        })
        out_path = PROCESSED_DIR / f"{name}.csv"
        out_df.to_csv(out_path, index=False)
        saved_files.append(out_path)
        logger.info(f"Saved {name}.csv ({len(out_df)} hours, mean: {series.mean():.1f} vCPUs, peak: {series.max():.1f} vCPUs)")

    return {"saved_files": saved_files}


def calibrate_and_generate_alibaba_workloads() -> Path:
    """
    Processes Alibaba cluster telemetry and generates a 30-day (720-hour)
    cross-validation benchmark trace.
    """
    raw_path = RAW_DIR / "alibaba_machine_usage.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw Alibaba dataset at {raw_path}. Run alibaba_loader.py first.")

    logger.info("Calibrating on Alibaba cluster trace...")
    df = pd.read_csv(raw_path)

    emp_mean_cpu = float(df["cpu_util_percent"].mean())
    machines = 200
    vcpu_per_machine = 4
    base_ali_demand = machines * vcpu_per_machine * (emp_mean_cpu / 100.0)

    hours = np.arange(HORIZON_HOURS)
    np.random.seed(99)

    # Alibaba batch workload: bursty with diurnal underlying traffic
    diurnal = 0.25 * base_ali_demand * np.sin(2 * np.pi * (hours - 10) / 24)
    batch_bursts = (np.random.rand(HORIZON_HOURS) > 0.92) * np.random.uniform(base_ali_demand * 0.7, base_ali_demand * 2.0, size=HORIZON_HOURS)
    noise = np.random.normal(0, base_ali_demand * 0.08, size=HORIZON_HOURS)
    ali_series = np.maximum(base_ali_demand * 0.75 + diurnal + batch_bursts + noise, 20.0)

    start_timestamp = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [start_timestamp + pd.Timedelta(hours=int(h)) for h in hours]

    out_df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_vcpu": np.round(ali_series, 2),
        "workload_type": "alibaba_bursty"
    })
    out_path = PROCESSED_DIR / "alibaba_bursty.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved alibaba_bursty.csv ({len(out_df)} hours, mean: {ali_series.mean():.1f} vCPUs, peak: {ali_series.max():.1f} vCPUs)")
    return out_path


def run_all():
    logger.info("=== Starting Workload Preprocessing Pipeline ===")
    calibrate_and_generate_azure_workloads()
    calibrate_and_generate_alibaba_workloads()
    logger.info("=== All 5 Workload Traces Successfully Generated in data/processed/workload_traces/ ===")


if __name__ == "__main__":
    run_all()
