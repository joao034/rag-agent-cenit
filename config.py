"""
Configuración del agente: constantes, credenciales y carga del índice.

Todo lo que se necesita UNA sola vez, al arrancar el proceso, vive aquí:
los clientes de Groq y Gemini, y el índice de fragmentos ya vectorizado.
"""

import json
import os
import pathlib

import numpy as np
from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------
# Constantes del modelo y del agente
# --------------------------------------------------------------------------------

MODELO = "openai/gpt-oss-120b"
MODELO_EMBEDDINGS = "gemini-embedding-001"
DIMENSIONES = 768

# Si el mejor fragmento no llega a este parecido, decimos que no sabemos.
# El número está medido, no adivinado: con 10 preguntas dentro del corpus y 8 fuera,
# las de dentro puntuaron entre 0.718 y 0.813, y las claramente ajenas por debajo de
# 0.66. En medio quedan las preguntas que SUENAN a Instituto pero no están escritas
# en ningún documento ("¿tienen convenio con la UNAM?" da 0.714). Ningún umbral las
# separa bien; preferimos dejarlas pasar y que el modelo, que ve la fuente y el score
# de cada fragmento, diga que eso no está documentado.
UMBRAL_SIMILITUD = 0.68

MAX_VUELTAS = 6
IVA = 0.16

INSTRUCCIONES = (
    "Eres el asistente del Instituto Cénit. Respondes preguntas sobre cursos, precios, "
    "becas y disponibilidad. Usa siempre las herramientas: nunca inventes precios, "
    "requisitos ni cupos. Si necesitas varios datos, pide varias herramientas. "
    "Responde en español, de forma breve y concreta."
)

# --------------------------------------------------------------------------------
# Clientes de las APIs externas
# --------------------------------------------------------------------------------

cliente = Groq(api_key=os.environ["GROQ_API_KEY"])
cliente_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --------------------------------------------------------------------------------
# Índice de embeddings — se carga UNA vez, cuando arranca el proceso.
# Son ~200 KB de JSON: leerlo por petición añadiría cientos de milisegundos gratis.
# --------------------------------------------------------------------------------

ARCHIVO_INDICE = pathlib.Path(__file__).parent / "index.json"
INDICE = json.loads(ARCHIVO_INDICE.read_text(encoding="utf-8"))
MATRIZ = np.array([f["vector"] for f in INDICE["fragmentos"]], dtype=np.float32)