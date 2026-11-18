"""
Module: azure_loader
Purpose: Download and load Microsoft Azure Public Dataset (VM CPU readings) 
         directly from official Microsoft GitHub Releases into data/raw/ as CSV.
"""

import os
import gzip
import shutil
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Official Microsoft Azure Public Dataset Release URL (File 1 of 125, ~35MB compressed)
AZURE_V1_FILE1_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v1/"
    "trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-125.csv.gz"
)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with streaming and progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {url}")
    logger.info(f"Target destination: {dest_path}")

    # Use urllib with browser-like user agent
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, "wb") as out_file:
        total_size = response.getheader("Content-Length")
        total_mb = f"{int(total_size) / (1024 * 1024):.1f} MB" if total_size else "unknown size"
        logger.info(f"Content length: {total_mb}")

        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            mb_downloaded = downloaded / (1024 * 1024)
            if total_size:
                percent = (downloaded / int(total_size)) * 100
                print(f"\rDownloading: {mb_downloaded:.1f} MB / {total_mb} ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\rDownloading: {mb_downloaded:.1f} MB", end="", flush=True)
        print()

    logger.info("Download completed successfully.")
    return dest_path


def decompress_gz_to_csv(gz_path: Path, csv_path: Path, max_rows: int = None) -> Path:
    """
    Decompresses a .gz file to .csv.
    Adds header if the raw trace has no header: timestamp,vm_id,min_cpu,max_cpu,avg_cpu
    """
    logger.info(f"Decompressing {gz_path.name} -> {csv_path.name}...")
    headers = "timestamp,vm_id,min_cpu,max_cpu,avg_cpu\n"

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as gz_in, open(csv_path, "w", encoding="utf-8") as csv_out:
        # Check first line to see if header already exists
        first_line = gz_in.readline()
        if not first_line.startswith("timestamp"):
            csv_out.write(headers)
        csv_out.write(first_line)

        row_count = 1
        for line in gz_in:
            csv_out.write(line)
            row_count += 1
            if max_rows and row_count >= max_rows:
                logger.info(f"Reached requested limit of {max_rows:,} rows.")
                break

    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    logger.info(f"Decompression finished: {row_count:,} rows saved ({file_size_mb:.1f} MB) to {csv_path}")
    return csv_path


def load_azure_dataset(max_rows: int = None, force_download: bool = False) -> Path:
    """
    Orchestrates downloading and saving the raw Azure VM dataset to CSV.
    """
    gz_filename = "azure_vm_cpu_readings_file1.csv.gz"
    csv_filename = "azure_vm_cpu_readings.csv"

    gz_path = RAW_DATA_DIR / gz_filename
    csv_path = RAW_DATA_DIR / csv_filename

    # Step 1: Download if not present
    if not gz_path.exists() or force_download:
        download_file(AZURE_V1_FILE1_URL, gz_path)
    else:
        logger.info(f"Found cached archive: {gz_path}")

    # Step 2: Decompress to CSV if not present
    if not csv_path.exists() or force_download:
        decompress_gz_to_csv(gz_path, csv_path, max_rows=max_rows)
    else:
        logger.info(f"Found existing raw CSV: {csv_path}")

    return csv_path


if __name__ == "__main__":
    logger.info("=== Starting Azure Public Dataset Loader ===")
    output_csv = load_azure_dataset()
    logger.info(f"=== Azure Dataset Ready at: {output_csv} ===")
