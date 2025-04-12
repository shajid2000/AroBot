from django.urls import path
from .views import InvokeAgentView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("invoke/", InvokeAgentView.as_view(), name="invoke"),
]

urlpatterns.append(static(settings.STATIC_URL, document_root=settings.STATIC_ROOT))
