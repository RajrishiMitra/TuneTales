from django.contrib import admin
from django.urls import path
from . import views
from . import camera

urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("recommendation/", views.recommendation, name="input_data"),
    path('camera/',camera.capture_emotion, name='camera')  # Add this line to include the camera URL
]
