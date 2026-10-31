
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone


def home(request):
    return render(request, "dashboard/index.html")


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
            "carbon_collector": "active",
            "scheduler": "initializing"
        }
    })
