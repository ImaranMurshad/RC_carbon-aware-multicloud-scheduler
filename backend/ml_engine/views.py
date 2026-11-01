
import csv
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.http import require_GET


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROLLING_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "ml"
    / "rolling_validation_summary.csv"
)


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
def forecasting_results(request):
    rows = read_csv_file(ROLLING_SUMMARY_FILE)

    if rows is None:
        return JsonResponse(
            {
                "status": "error",
                "message": "Rolling validation results file not found."
            },
            status=503
        )

    results = convert_numeric_values(rows)

    return JsonResponse({
        "status": "success",
        "evaluation": "Four-fold expanding-window rolling validation",
        "primary_metric": "MAE",
        "total_records": len(results),
        "results": results
    })
