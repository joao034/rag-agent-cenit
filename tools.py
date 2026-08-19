"""
Las tres herramientas que el agente puede llamar.

Cada función hace una sola cosa y devuelve texto plano — es lo que el
modelo va a leer como resultado de la herramienta. Al final del archivo
está el esquema (para el function-calling de Groq) y el diccionario que
mapea nombre -> función, que usa el loop del agente.
"""

import numpy as np
from google.genai import types

import data
from config import (
    DIMENSIONES,
    IVA,
    INDICE,
    MATRIZ,
    MODELO_EMBEDDINGS,
    UMBRAL_SIMILITUD,
    cliente_gemini,
)


def buscar_en_corpus(consulta):
    """Busca en los documentos del Instituto y devuelve los 3 fragmentos más parecidos."""
    respuesta = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDINGS,
        contents=consulta,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONES,
        ),
    )
    vector = np.array(respuesta.embeddings[0].values, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)

    similitudes = MATRIZ @ vector
    mejores = np.argsort(-similitudes)[:3]

    if similitudes[mejores[0]] < UMBRAL_SIMILITUD:
        return "No encontré información sobre eso en los documentos del Instituto Cénit."

    partes = []
    for posicion in mejores:
        fragmento = INDICE["fragmentos"][posicion]
        partes.append(
            f"[fuente: {fragmento['fuente']} — {fragmento['titulo']} "
            f"| parecido: {similitudes[posicion]:.2f}]\n{fragmento['texto']}"
        )
    return "\n\n".join(partes)


def calcular_costo(codigo_curso, modalidad, tipo_beca=None):
    """Devuelve el desglose del costo de un curso, línea por línea."""
    codigo = codigo_curso.strip().upper()
    if codigo not in data.PRECIOS:
        return f"No existe el curso {codigo}. Códigos válidos: {', '.join(data.PRECIOS)}."

    texto_modalidad = modalidad.strip().lower().replace("í", "i")
    if "presencial" in texto_modalidad:
        clave_modalidad, etiqueta_modalidad = "presencial", "presencial"
    elif "linea" in texto_modalidad or "online" in texto_modalidad or "virtual" in texto_modalidad:
        clave_modalidad, etiqueta_modalidad = "linea", "en línea"
    else:
        return f"Modalidad '{modalidad}' no reconocida. Usa 'presencial' o 'en línea'."

    curso = data.PRECIOS[codigo]
    base = curso[clave_modalidad]

    lineas = [
        f"{codigo} — {curso['nombre']} ({etiqueta_modalidad})",
        f"  {'Precio base':<34} ${base:>9,.2f}",
    ]

    descuento = 0.0
    if tipo_beca:
        texto_beca = tipo_beca.strip().lower()
        clave_beca = next((k for k in data.BECAS if k in texto_beca or texto_beca in k), None)
        if clave_beca is None:
            return f"No existe la beca '{tipo_beca}'. Becas válidas: {', '.join(data.BECAS)}."
        beca = data.BECAS[clave_beca]
        if beca["solo_en_linea"] and clave_modalidad != "linea":
            lineas.append(
                f"  {beca['nombre']}: NO APLICA — solo es válida en modalidad en línea."
            )
        else:
            descuento = base * beca["porcentaje"]
            etiqueta_beca = f"{beca['nombre']} ({beca['porcentaje']:.0%})"
            lineas.append(f"  {etiqueta_beca:<34}-${descuento:>9,.2f}")

    subtotal = base - descuento
    impuesto = subtotal * IVA
    total = subtotal + impuesto

    lineas.append(f"  {'Subtotal':<34} ${subtotal:>9,.2f}")
    lineas.append(f"  {f'IVA ({IVA:.0%})':<34} ${impuesto:>9,.2f}")
    lineas.append("  " + "─" * 45)
    lineas.append(f"  {'TOTAL A PAGAR':<34} ${total:>9,.2f}")
    return "\n".join(lineas)


def consultar_cupo(codigo_curso):
    """Consulta los lugares disponibles. Simula un servicio externo del Instituto."""
    # Se lee data.ESCENARIO (no se importa el valor directo) para que, si alguien
    # lo cambia en vivo en data.py durante una demo con --reload, el cambio se vea
    # de inmediato sin tener que tocar este archivo.
    if data.ESCENARIO == "timeout":
        raise TimeoutError("El servicio de inscripciones no respondió (timeout de 5 s).")
    if data.ESCENARIO == "error500":
        raise RuntimeError("HTTP 500 del servicio de inscripciones. Intenta de nuevo.")

    codigo = codigo_curso.strip().upper()
    if codigo not in data.CUPOS:
        return f"No existe el curso {codigo}. Códigos válidos: {', '.join(data.CUPOS)}."

    cupo = data.CUPOS[codigo]
    if cupo["disponibles"] == 0:
        return (
            f"{codigo}: SIN CUPO. Los {cupo['total']} lugares están ocupados. "
            f"Hay lista de espera abierta."
        )
    return (
        f"{codigo}: {cupo['disponibles']} lugares disponibles de {cupo['total']}. "
        f"Inscripciones abiertas."
    )


ESQUEMA_HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_en_corpus",
            "description": (
                "Busca en los documentos oficiales del Instituto Cénit: catálogo de "
                "cursos, prerrequisitos, horarios, reglas de becas, requisitos para "
                "solicitarlas, formas de pago, reembolsos y preguntas frecuentes. "
                "Úsala para cualquier pregunta de reglas o requisitos. No sirve para "
                "calcular precios ni para consultar cupos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": (
                            "Qué quieres buscar, redactado como una frase completa. "
                            "Por ejemplo: 'requisitos para la beca de egresados'."
                        ),
                    }
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_costo",
            "description": (
                "Calcula el costo total de un curso, con el desglose línea por línea: "
                "precio base, descuento de beca, subtotal, IVA y total. Úsala siempre "
                "que te pregunten un precio o cuánto sale algo. Nunca calcules de memoria."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codigo_curso": {
                        "type": "string",
                        "description": "Código del curso, por ejemplo 'NBL-204'.",
                    },
                    "modalidad": {
                        "type": "string",
                        "enum": ["presencial", "en línea"],
                        "description": (
                            "Modalidad en la que se cursa. Si la persona no la menciona, "
                            "usa 'presencial'."
                        ),
                    },
                    "tipo_beca": {
                        "type": "string",
                        "enum": ["egresados", "excelencia", "comunidad"],
                        "description": (
                            "Beca que se quiere aplicar. Omite este parámetro si la "
                            "persona no menciona ninguna beca."
                        ),
                    },
                },
                "required": ["codigo_curso", "modalidad"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_cupo",
            "description": (
                "Consulta cuántos lugares quedan disponibles en un curso. Es el único "
                "dato que cambia día a día, así que consúltalo siempre que pregunten si "
                "hay lugar, si está lleno o si todavía se pueden inscribir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codigo_curso": {
                        "type": "string",
                        "description": "Código del curso, por ejemplo 'NBL-204'.",
                    }
                },
                "required": ["codigo_curso"],
            },
        },
    },
]

HERRAMIENTAS = {
    "buscar_en_corpus": buscar_en_corpus,
    "calcular_costo": calcular_costo,
    "consultar_cupo": consultar_cupo,
}