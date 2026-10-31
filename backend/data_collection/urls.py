from django.urls import path
from . import views

urlpatterns = [
    path("carbon/", views.carbon_readings, name="carbon_readings"),
]