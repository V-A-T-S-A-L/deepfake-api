from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_image, name='detect_image'),
    path('health/', views.api_health, name='api_health'),
]