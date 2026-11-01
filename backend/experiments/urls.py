from django.urls import path
from . import views

urlpatterns = [
    path(
        "results/",
        views.scheduler_experiment_results,
        name="scheduler_experiment_results"
    ),
]
