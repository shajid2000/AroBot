from django.urls import path
from .views import InvokeAgentView

urlpatterns = [
    path("invoke/", InvokeAgentView.as_view(), name="invoke"),
]
