
import os
import sys
import csv
import django

from pathlib import Path

# ==========================================
# CARBON-AWARE MULTI-CLOUD SCHEDULER
# Historical Carbon Data Export
# ==========================================

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DATASET_DIR = PROJECT_ROOT / "datasets"

# Configure Django
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

django.setup()

from data_collection.models import CarbonReading


def export_carbon_data():

    print("\n==========================================")
    print("HISTORICAL CARBON DATA EXPORT")
    print("==========================================")

    # Create datasets directory
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Output CSV file
    output_file = DATASET_DIR / "carbon_readings.csv"

    # Retrieve observations from PostgreSQL
    readings = CarbonReading.objects.all().order_by(
        "zone",
        "timestamp"
    )

    total_records = readings.count()

    if total_records == 0:
        print("No carbon readings found.")
        return

    # CSV columns
    fieldnames = [
        "zone",
        "carbon_intensity",
        "timestamp",
        "is_estimated",
    ]

    # Export records
    with open(
        output_file,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for reading in readings:

            writer.writerow({
                "zone": reading.zone,
                "carbon_intensity": reading.carbon_intensity,
                "timestamp": reading.timestamp.isoformat(),
                "is_estimated": reading.is_estimated,
            })

    # Export summary
    print(f"\nExport completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Total records exported: {total_records}")

    print("\nRecords per zone:")

    for zone in ["IE", "DE", "FR"]:
        count = readings.filter(zone=zone).count()
        print(f"{zone}: {count}")

    print("\n==========================================")
    print("EXPORT COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    export_carbon_data()
