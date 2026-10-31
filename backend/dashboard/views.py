from django.http import JsonResponse
from django.utils import timezone


def health_check(request):
    return JsonResponse({
        "project": "Carbon-Aware Multi-Cloud Scheduler",
        "status": "running",
        "message": "Backend is working successfully!",
        "timestamp": timezone.now().isoformat(),
        "services": {
            "django": "active",
            "database": "active",
            "ml_engine": "initializing",
            "carbon_collector": "initializing",
            "scheduler": "initializing"
        }
    })