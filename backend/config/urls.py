
from django.contrib import admin
from django.urls import path, include
from dashboard.views import home

urlpatterns = [
    path("", home, name="home"),
    path("admin/", admin.site.urls),
    path("api/", include("dashboard.urls")),
    path("api/", include("data_collection.urls")),
]
