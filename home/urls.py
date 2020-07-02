from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name='home'),
    path('tag',views.result,name='result'),
    path('Charts',views.Charts,name='result1')
    

]
