
import os
import sys
import django

from datetime import timedelta
from django.db.models import Min, Max, Avg, Count

# Configure Django
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

sys.path.insert(0, BACKEND_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

django.setup()

from data_collection.models import CarbonReading


ZONES = ["IE", "DE", "FR"]
EXPECTED_INTERVAL = timedelta(minutes=15)


def run_data_quality_check():

    print("\n==========================================")
    print("CARBON DATA QUALITY REPORT")
    print("==========================================")

    total_records = CarbonReading.objects.count()
    print(f"\nTotal database records: {total_records}")

    for zone in ZONES:

        readings = CarbonReading.objects.filter(
            zone=zone
        ).order_by("timestamp")

        timestamps = list(
            readings.values_list("timestamp", flat=True)
        )

        statistics = readings.aggregate(
            count=Count("id"),
            first=Min("timestamp"),
            last=Max("timestamp"),
            average=Avg("carbon_intensity"),
        )

        missing_timestamps = []

        for previous, current in zip(
            timestamps, timestamps[1:]
        ):

            difference = current - previous

            if difference > EXPECTED_INTERVAL:

                missing = previous + EXPECTED_INTERVAL

                while missing < current:
                    missing_timestamps.append(missing)
                    missing += EXPECTED_INTERVAL

        estimated_count = readings.filter(
            is_estimated=True
        ).count()

        print(f"\nZONE: {zone}")
        print("------------------------------------------")
        print(f"Records: {statistics['count']}")
        print(f"First reading: {statistics['first']}")
        print(f"Latest reading: {statistics['last']}")
        print(f"Average intensity: {statistics['average']}")
        print(f"Estimated readings: {estimated_count}")
        print(f"Missing timestamps: {len(missing_timestamps)}")

        for timestamp in missing_timestamps:
            print(f"  - {timestamp}")

    print("\n==========================================")
    print("DATA QUALITY CHECK COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    run_data_quality_check()
