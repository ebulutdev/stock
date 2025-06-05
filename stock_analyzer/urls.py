from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze_stock/', views.analyze_stock, name='analyze_stock'),
] 