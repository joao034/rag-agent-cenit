"""
Agente del Instituto Cénit — punto de entrada de la API.

Este archivo solo define la app de FastAPI y sus tres endpoints. La lógica
del agente vive en agent.py, las herramientas en tools.py, los datos de
negocio en data.py y la configuración/credenciales en config.py.

Para correrlo en tu máquina:
    export GROQ_API_KEY=...
    export GEMINI_API_KEY=...
    uvicorn main:app --reload
"""

import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import correr_agente

app = FastAPI(title="Agente Instituto Cénit")

# Abierto a propósito: el cliente de Streamlit corre en localhost, en otra máquina.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUTA_PAGINA = pathlib.Path(__file__).parent / "static" / "index.html"


class Peticion(BaseModel):
    pregunta: str


class Respuesta(BaseModel):
    respuesta: str
    trayectoria: list
    vueltas: int


@app.post("/preguntar", response_model=Respuesta)
def preguntar(peticion: Peticion):
    """Le pasa la pregunta al agente y devuelve también CÓMO llegó a la respuesta."""
    contenido, trayectoria, vueltas = correr_agente(peticion.pregunta)
    return Respuesta(respuesta=contenido, trayectoria=trayectoria, vueltas=vueltas)


@app.get("/health")
def health():
    """Render pega aquí para saber si el servicio sigue vivo."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def inicio():
    """Una página mínima para probar el agente desde el navegador, sin instalar nada."""
    return RUTA_PAGINA.read_text(encoding="utf-8")