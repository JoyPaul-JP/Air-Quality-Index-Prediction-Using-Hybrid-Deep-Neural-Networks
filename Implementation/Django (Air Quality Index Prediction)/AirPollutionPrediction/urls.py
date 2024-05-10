from django.contrib import admin
from django.urls import path, include
from PredictApp import views  # Import views from your app
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),  # Route root URL to the index view
    path('index/', include('PredictApp.urls')),  # Include other app URLs
]

# Serving static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)