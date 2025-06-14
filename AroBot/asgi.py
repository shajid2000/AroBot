"""
ASGI config for AroBot project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.routing import URLRouter
from jarvis import routing as jarvis_routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AroBot.settings')

websocket_urlpatterns = []
websocket_urlpatterns.extend(jarvis_routing.websocket_urlpatterns)

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})
