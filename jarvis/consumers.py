import asyncio
import base64
import json
from typing import AsyncIterable
from channels.generic.websocket import AsyncWebsocketConsumer
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from jarvis.agent import root_agent  # Make sure this import path is correct

APP_NAME = "Jarvis ADK Streaming"
session_service = InMemorySessionService()

class JarvisConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = None
        self.live_events = None
        self.live_request_queue = None
        self.agent_to_client_task = None
        self.is_audio = False

    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        
        # Get is_audio parameter from query string
        query_string = self.scope.get('query_string', b'').decode()
        query_params = {}
        if query_string:
            query_params = dict(param.split('=') for param in query_string.split('&') if '=' in param)
        self.is_audio = query_params.get('is_audio', 'false') == 'true'
        
        await self.accept()
        print(f"Jarvis Client #{self.session_id} connected, audio mode: {self.is_audio}")
        
        try:
            # Start agent session
            self.live_events, self.live_request_queue = await self.start_agent_session(
                self.session_id, self.is_audio
            )
            
            # Start agent to client messaging task
            self.agent_to_client_task = asyncio.create_task(
                self.agent_to_client_messaging()
            )
        except Exception as e:
            print(f"Error starting agent session: {e}")
            await self.close(code=4000)

    async def disconnect(self, close_code):
        if self.agent_to_client_task:
            self.agent_to_client_task.cancel()
        print(f"Jarvis Client #{self.session_id} disconnected")

    async def receive(self, text_data):
        """Handle incoming messages from client"""
        try:
            message = json.loads(text_data)
            await self.client_to_agent_messaging(message)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            await self.send(text_data=json.dumps({
                "error": "Invalid JSON format"
            }))
        except Exception as e:
            print(f"Error handling message: {e}")
            await self.send(text_data=json.dumps({
                "error": f"Message handling error: {str(e)}"
            }))

    async def start_agent_session(self, session_id, is_audio=False):
        """Starts an agent session"""
        try:
            # Create a Session
            session = await session_service.create_session(
                app_name=APP_NAME,
                user_id=session_id,
                session_id=session_id,
            )

            # Create a Runner
            runner = Runner(
                app_name=APP_NAME,
                agent=root_agent,
                session_service=session_service,
            )

            # Set response modality
            modality = "AUDIO" if is_audio else "TEXT"

            # Create speech config with voice settings
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            )

            # Create run config with basic settings
            config = {"response_modalities": [modality], "speech_config": speech_config}

            # Add output_audio_transcription when audio is enabled to get both audio and text
            if is_audio:
                config["output_audio_transcription"] = {}

            run_config = RunConfig(**config)

            # Create a LiveRequestQueue for this session
            live_request_queue = LiveRequestQueue()

            # Start agent session
            live_events = runner.run_live(
                session=session,
                live_request_queue=live_request_queue,
                run_config=run_config,
            )
            return live_events, live_request_queue
        except Exception as e:
            print(f"Error in start_agent_session: {e}")
            raise

    async def agent_to_client_messaging(self):
        """Agent to client communication"""
        try:
            async for event in self.live_events:
                if event is None:
                    continue

                # If the turn complete or interrupted, send it
                if event.turn_complete or event.interrupted:
                    message = {
                        "turn_complete": event.turn_complete,
                        "interrupted": event.interrupted,
                    }
                    await self.send(text_data=json.dumps(message))
                    print(f"[JARVIS AGENT TO CLIENT]: {message}")
                    continue

                # Read the Content and its first Part
                part = event.content and event.content.parts and event.content.parts[0]
                if not part:
                    continue

                # Make sure we have a valid Part
                if not isinstance(part, types.Part):
                    continue

                # Only send text if it's a partial response (streaming)
                # Skip the final complete message to avoid duplication
                if part.text and event.partial:
                    message = {
                        "mime_type": "text/plain",
                        "data": part.text,
                        "role": "model",
                    }
                    await self.send(text_data=json.dumps(message))
                    print(f"[JARVIS AGENT TO CLIENT]: text/plain: {part.text}")

                # If it's audio, send Base64 encoded audio data
                is_audio = (
                    part.inline_data
                    and part.inline_data.mime_type
                    and part.inline_data.mime_type.startswith("audio/pcm")
                )
                if is_audio:
                    audio_data = part.inline_data and part.inline_data.data
                    if audio_data:
                        message = {
                            "mime_type": "audio/pcm",
                            "data": base64.b64encode(audio_data).decode("ascii"),
                            "role": "model",
                        }
                        await self.send(text_data=json.dumps(message))
                        print(f"[JARVIS AGENT TO CLIENT]: audio/pcm: {len(audio_data)} bytes.")
        except asyncio.CancelledError:
            print("Jarvis agent to client messaging task cancelled")
        except Exception as e:
            print(f"Error in jarvis agent to client messaging: {e}")
            await self.send(text_data=json.dumps({
                "error": f"Agent messaging error: {str(e)}"
            }))

    async def client_to_agent_messaging(self, message):
        """Client to agent communication"""
        try:
            mime_type = message.get("mime_type")
            data = message.get("data")
            role = message.get("role", "user")  # Default to 'user' if role is not provided

            if not mime_type or not data:
                raise ValueError("Missing mime_type or data in message")

            # Send the message to the agent
            if mime_type == "text/plain":
                # Send a text message
                content = types.Content(role=role, parts=[types.Part.from_text(text=data)])
                self.live_request_queue.send_content(content=content)
                print(f"[JARVIS CLIENT TO AGENT]: {data}")
            elif mime_type == "audio/pcm":
                # Send audio data
                decoded_data = base64.b64decode(data)

                # Send the audio data - note that ActivityStart/End and transcription
                # handling is done automatically by the ADK when input_audio_transcription
                # is enabled in the config
                self.live_request_queue.send_realtime(
                    types.Blob(data=decoded_data, mime_type=mime_type)
                )
                print(f"[JARVIS CLIENT TO AGENT]: audio/pcm: {len(decoded_data)} bytes")
            else:
                raise ValueError(f"Mime type not supported: {mime_type}")
        except Exception as e:
            print(f"Error in jarvis client to agent messaging: {e}")
            await self.send(text_data=json.dumps({
                "error": f"Client messaging error: {str(e)}"
            }))
