"""
Datos de negocio del Instituto: precios, becas y cupos.

Separados del resto del código porque son los datos que más cambian
(sobre todo CUPOS y ESCENARIO) y conviene poder tocarlos sin tocar
la lógica de las herramientas.
"""

PRECIOS = {
    "CNT-101": {"nombre": "Fundamentos de IA Aplicada", "presencial": 380, "linea": 280},
    "CNT-204": {"nombre": "Recuperación Aumentada (RAG) en Producción", "presencial": 520, "linea": 400},
    "CNT-210": {"nombre": "Agentes Autónomos con Herramientas", "presencial": 600, "linea": 460},
    "CNT-305": {"nombre": "Evaluación y Observabilidad de Sistemas LLM", "presencial": 490, "linea": 360},
    "CNT-402": {"nombre": "Despliegue y Costos de Modelos en la Nube", "presencial": 680, "linea": 540},
}

BECAS = {
    "egresados": {"nombre": "Beca Egresado Cénit", "porcentaje": 0.25, "solo_en_linea": False},
    "excelencia": {"nombre": "Beca de Excelencia Académica", "porcentaje": 0.40, "solo_en_linea": False},
    "comunidad": {"nombre": "Beca Comunidad Abierta", "porcentaje": 0.15, "solo_en_linea": True},
}

# En clase esta constante se cambia en vivo para ver fallar al agente.
# En el servidor desplegado se queda en "ok".
ESCENARIO = "ok"  # "ok" | "timeout" | "error500"

CUPOS = {
    "CNT-101": {"disponibles": 12, "total": 30},
    "CNT-204": {"disponibles": 3, "total": 25},
    "CNT-210": {"disponibles": 0, "total": 28},
    "CNT-305": {"disponibles": 8, "total": 20},
    "CNT-402": {"disponibles": 5, "total": 15},
}