
import csv
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.http import require_GET


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

SUMMARY_FILE = RESULTS_DIR / "scheduler_strategy_summary.csv"
SELECTIONS_FILE = RESULTS_DIR / "scheduler_provider_selections.csv"


def read_csv_file(file_path):
    if not file_path.exists():
        return None

    with file_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def convert_numeric_values(rows):
    converted_rows = []

    for row in rows:
        converted = {}

        for key, value in row.items():
            if value == "":
                converted[key] = None
                continue

            try:
                converted[key] = float(value)
            except (ValueError, TypeError):
                converted[key] = value

        converted_rows.append(converted)

    return converted_rows


@require_GET
def scheduler_experiment_results(request):
    summary = read_csv_file(SUMMARY_FILE)
    selections = read_csv_file(SELECTIONS_FILE)

    if summary is None or selections is None:
        return JsonResponse(
            {
                "status": "error",
                "message": "One or more experiment result files are missing."
            },
            status=503
        )

    return JsonResponse({
        "status": "success",
        "experiment": "Historical multi-strategy scheduling comparison",
        "data_type": "Simulated scheduling experiment",
        "assumptions": [
            "Provider costs are assumed values.",
            "Provider latencies are assumed values.",
            "Provider resource capacities are assumed values.",
            "Carbon intensity is a regional proxy, not measured workload emissions."
        ],
        "summary": convert_numeric_values(summary),
        "provider_selections": convert_numeric_values(selections)
    })
