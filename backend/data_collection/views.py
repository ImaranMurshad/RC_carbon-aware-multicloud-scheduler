from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import CarbonReading


@api_view(["GET"])
def carbon_readings(request):
    readings = CarbonReading.objects.order_by("-timestamp")

    data = [
        {
            "zone": reading.zone,
            "carbon_intensity": float(reading.carbon_intensity),
            "timestamp": reading.timestamp.isoformat(),
            "updated_at": (
                reading.updated_at.isoformat()
                if reading.updated_at else None
            ),
            "is_estimated": reading.is_estimated,
            "unit": "gCO2e/kWh",
        }
        for reading in readings
    ]

    return Response({
        "count": len(data),
        "readings": data,
    })