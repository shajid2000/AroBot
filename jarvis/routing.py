from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/jarvis/(?P<session_id>\w+)/$', consumers.JarvisConsumer.as_asgi()),
]