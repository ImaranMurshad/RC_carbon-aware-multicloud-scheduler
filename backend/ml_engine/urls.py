from django.urls import path
from . import views

urlpatterns = [
    path(
        "forecasting/",
        views.forecasting_results,
        name="forecasting_results"
    ),
]