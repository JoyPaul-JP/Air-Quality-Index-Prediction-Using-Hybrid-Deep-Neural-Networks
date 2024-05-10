from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.index, name='index'),
    path('signin/', views.signin, name='signin'),
    path('dashboard/', views.dashboard, name='dashboard'),  # Define URL pattern for the dashboard
    path('logout/', views.logout, name='logout'),
    path('generate-pdf/', views.generate_pdf, name='generate_pdf'),
    
]
