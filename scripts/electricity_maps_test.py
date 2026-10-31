import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("ELECTRICITY_MAPS_API_KEY")

ZONES = ["IE", "DE", "FR"]

API_URL = "https://api.electricitymaps.com/v4/carbon-intensity/latest"


def fetch_carbon_intensity(zone):
    headers = {"auth-token": API_KEY}
    params = {
        "zone": zone,
        "temporalGranularity": "15_minutes",
        "emissionFactorType": "lifecycle",
        "flowTraced": "true",
    }

    try:
        response = requests.get(
            API_URL,
            headers=headers,
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        return {"zone": zone, "success": True, "data": data}

    except requests.RequestException as error:
        return {"zone": zone, "success": False, "error": str(error)}


def main():
    if not API_KEY:
        print("ERROR: API key not found. Check your .env file.")
        return

    print("=" * 60)
    print("ELECTRICITY MAPS - LIVE CARBON INTENSITY")
    print("Collection started:", datetime.now().astimezone())
    print("=" * 60)

    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fetch_carbon_intensity, zone): zone
            for zone in ZONES
        }

        for future in as_completed(futures):
            result = future.result()
            zone = result["zone"]

            print(f"\nREGION: {zone}")

            if result["success"]:
                data = result["data"]

                print(
                    "Carbon Intensity:",
                    data.get("carbonIntensity"),
                    "gCO2e/kWh",
                )
                print("Timestamp:", data.get("datetime"))
                print("Updated At:", data.get("updatedAt"))
                print("Estimated:", data.get("isEstimated"))
                print("Granularity:", data.get("temporalGranularity"))
            else:
                print("Request failed:", result["error"])

    duration = (datetime.now() - start_time).total_seconds()
    print("\nCollection completed in", round(duration, 2), "seconds.")


if __name__ == "__main__":
    main()