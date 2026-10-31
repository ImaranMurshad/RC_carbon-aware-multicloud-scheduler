import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import requests
from dotenv import load_dotenv
from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_datetime

from data_collection.models import CarbonReading


load_dotenv()

API_URL = "https://api.electricitymaps.com/v4/carbon-intensity/latest"

ZONES = ["IE", "DE", "FR"]


def fetch_carbon_data(zone, api_key):
    headers = {"auth-token": api_key}

    params = {
        "zone": zone,
        "temporalGranularity": "15_minutes",
        "emissionFactorType": "lifecycle",
        "flowTraced": "true",
    }

    response = requests.get(
        API_URL,
        headers=headers,
        params=params,
        timeout=20,
    )

    response.raise_for_status()
    return response.json()


class Command(BaseCommand):
    help = "Collect carbon intensity readings from Electricity Maps"

    def handle(self, *args, **options):
        api_key = os.getenv("ELECTRICITY_MAPS_API_KEY")

        if not api_key:
            self.stdout.write(
                self.style.ERROR(
                    "Electricity Maps API key not found. Check your .env file."
                )
            )
            return

        self.stdout.write(
            self.style.SUCCESS("Starting carbon data collection...")
        )

        saved_count = 0

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(fetch_carbon_data, zone, api_key): zone
                for zone in ZONES
            }

            for future in as_completed(futures):
                zone = futures[future]

                try:
                    data = future.result()

                    intensity = data.get("carbonIntensity")
                    timestamp = parse_datetime(data.get("datetime", ""))
                    updated_at = parse_datetime(data.get("updatedAt", ""))

                    if intensity is None or timestamp is None:
                        raise ValueError(
                            "API response is missing carbon intensity or timestamp."
                        )

                    reading, created = CarbonReading.objects.update_or_create(
                        zone=zone,
                        timestamp=timestamp,
                        defaults={
                            "carbon_intensity": Decimal(str(intensity)),
                            "updated_at": updated_at,
                            "is_estimated": data.get("isEstimated", False),
                            "temporal_granularity": data.get(
                                "temporalGranularity", "15_minutes"
                            ),
                            "raw_response": data,
                        },
                    )

                    saved_count += 1

                    action = "CREATED" if created else "UPDATED"

                    self.stdout.write(
                        self.style.SUCCESS(
                            f"{zone}: {reading.carbon_intensity} "
                            f"gCO2e/kWh - {action}"
                        )
                    )

                except Exception as error:
                    self.stdout.write(
                        self.style.ERROR(f"{zone}: Collection failed - {error}")
                    )

        self.stdout.write(
            self.style.SUCCESS(
                f"Collection finished. Successfully saved/updated "
                f"{saved_count} zone readings."
            )
        )