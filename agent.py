"""
El loop del agente.

Le pasa la pregunta al modelo, ejecuta las herramientas que pida, y repite
hasta que el modelo responda sin pedir más herramientas o hasta agotar
MAX_VUELTAS. Todo lo que pueda tronar (argumentos mal formados, herramienta
inexistente, servicio caído) se atrapa y se convierte en texto, para que el
modelo pueda leer el error y decidir qué hacer.
"""

import json

from config import INSTRUCCIONES, MAX_VUELTAS, MODELO, cliente
from tools import ESQUEMA_HERRAMIENTAS, HERRAMIENTAS


def correr_agente(pregunta):
    """Corre el loop y devuelve (respuesta, trayectoria, vueltas)."""
    mensajes = [
        {"role": "system", "content": INSTRUCCIONES},
        {"role": "user", "content": pregunta},
    ]
    trayectoria = []
    vuelta = 0

    while vuelta < MAX_VUELTAS:
        vuelta += 1

        respuesta = cliente.chat.completions.create(
            model=MODELO,
            messages=mensajes,
            tools=ESQUEMA_HERRAMIENTAS,
            temperature=0,
            seed=42,
            include_reasoning=False,
        )
        mensaje = respuesta.choices[0].message

        mensajes.append(
            {
                "role": "assistant",
                "content": mensaje.content,
                "tool_calls": [
                    {
                        "id": llamada.id,
                        "type": "function",
                        "function": {
                            "name": llamada.function.name,
                            "arguments": llamada.function.arguments,
                        },
                    }
                    for llamada in (mensaje.tool_calls or [])
                ],
            }
        )

        if not mensaje.tool_calls:
            return mensaje.content, trayectoria, vuelta

        for llamada in mensaje.tool_calls:
            nombre = llamada.function.name

            # Todo lo que pueda tronar, truena aquí adentro y sale como texto:
            # argumentos mal formados, herramienta inexistente, servicio caído.
            try:
                argumentos = json.loads(llamada.function.arguments)
                resultado = HERRAMIENTAS[nombre](**argumentos)
            except Exception as error:
                argumentos = {"_sin_parsear": llamada.function.arguments}
                resultado = f"ERROR al ejecutar {nombre}: {type(error).__name__}: {error}"

            trayectoria.append(
                {"vuelta": vuelta, "herramienta": nombre, "argumentos": argumentos}
            )
            mensajes.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada.id,
                    "name": nombre,
                    "content": resultado,
                }
            )

    # Se acabaron las vueltas: una última llamada sin herramientas, para abstenerse bien.
    mensajes.append(
        {
            "role": "user",
            "content": (
                "Ya no puedes usar más herramientas. Responde con la información que SÍ "
                "lograste obtener y di claramente qué dato no pudiste conseguir y por qué. "
                "No inventes el dato que falta."
            ),
        }
    )
    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        temperature=0,
        seed=42,
        include_reasoning=False,
    )
    return respuesta.choices[0].message.content, trayectoria, vuelta