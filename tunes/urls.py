from django.contrib import admin
from django.urls import path
from . import views
from .views import cam_view

urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("recommendation/", views.recommendation, name="input_data"),
    path('cam/', views.cam_view, name='cam'),
]
