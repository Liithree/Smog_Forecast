from django.urls import path
from . import views,search

urlpatterns = [
    path('homepage/', views.homepage),
    path('historyDataView/', search.search_historyData),
    path('forecastData/', search.search_ForecastData),
]