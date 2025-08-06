from django.shortcuts import render

# Create your views here.

# chat/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag import get_chain
import uuid

class ChatView(APIView):
    """
    POST /api/chat/
    Body: { "message": "¿Cómo tramito mi CURP?", "conversation_id": "<opcional>" }
    """
    def post(self, request):
        user_msg = request.data.get("message")
        convo_id = request.data.get("conversation_id") or str(uuid.uuid4())

        if not user_msg:
            return Response({"error": "Falta 'message'."}, status=status.HTTP_400_BAD_REQUEST)

        chain = get_chain(convo_id)
        respuesta = chain(user_msg)["answer"]  # langchain devuelve 'answer'

        return Response({
            "conversation_id": convo_id,
            "response": respuesta
        })

