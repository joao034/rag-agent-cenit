"""
Genera index.json a partir de los documentos en docs/.

Esto se corre UNA sola vez. El resultado (index.json) se commitea al repo, y a partir
de ahí ni la clase ni el servidor desplegado vuelven a necesitar embebir los documentos:
solo embeben la pregunta del usuario.

    python create_index.py
"""

import json
import os
import pathlib

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- Configuración -----------------------------------------------------------------

MODELO_EMBEDDINGS = "gemini-embedding-001"
DIMENSIONES = 768  # el modelo da 3072 por defecto; 768 es más que suficiente y pesa menos

CARPETA_DOCS = pathlib.Path(__file__).parent / "docs"
ARCHIVO_SALIDA = pathlib.Path(__file__).parent / "index.json"


# --- Paso 1: partir los documentos en fragmentos -------------------------------------


def partir_en_fragmentos(texto_md, nombre_archivo):
    """
    Corta un markdown por sus encabezados de nivel 2 (##).

    Cada fragmento queda siendo una sección completa y autocontenida. No usamos ventanas
    de N caracteres a propósito: los documentos ya vienen organizados por tema, y cortar
    por tema da fragmentos que se entienden solos cuando el agente los cita.
    """
    fragmentos = []
    titulo_actual = None
    lineas_actuales = []

    for linea in texto_md.splitlines():
        if linea.startswith("## "):
            # empieza una sección nueva: guardamos la anterior
            if titulo_actual is not None:
                fragmentos.append((titulo_actual, "\n".join(lineas_actuales).strip()))
            titulo_actual = linea[3:].strip()
            lineas_actuales = []
        elif titulo_actual is not None:
            lineas_actuales.append(linea)

    if titulo_actual is not None:
        fragmentos.append((titulo_actual, "\n".join(lineas_actuales).strip()))

    # El texto que se embebe lleva el nombre del documento y el título de la sección.
    # Sin ese contexto, un fragmento como "Descuento del 25% sobre el precio" no se
    # parece a la pregunta "¿cuánto es la beca de egresados?".
    resultado = []
    for titulo, cuerpo in fragmentos:
        if not cuerpo:
            continue
        resultado.append(
            {
                "fuente": nombre_archivo,
                "titulo": titulo,
                "texto": f"[{nombre_archivo} — {titulo}]\n{cuerpo}",
            }
        )
    return resultado


print("Leyendo documentos de docs/ ...")
fragmentos = []
for archivo in sorted(CARPETA_DOCS.glob("*.md")):
    nuevos = partir_en_fragmentos(archivo.read_text(encoding="utf-8"), archivo.name)
    print(f"  {archivo.name}: {len(nuevos)} fragmentos")
    fragmentos.extend(nuevos)

print(f"Total: {len(fragmentos)} fragmentos\n")


# --- Paso 2: embeber cada fragmento --------------------------------------------------

cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def normalizar(vector):
    """
    gemini-embedding-001 solo devuelve vectores de norma 1 cuando pides las 3072
    dimensiones completas. Si truncas a 768, hay que normalizar a mano — si no, la
    similitud del coseno queda mal calculada.
    """
    v = np.array(vector, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


print(f"Embebiendo con {MODELO_EMBEDDINGS} ({DIMENSIONES} dimensiones)...")
for i, fragmento in enumerate(fragmentos, start=1):
    respuesta = cliente.models.embed_content(
        model=MODELO_EMBEDDINGS,
        contents=fragmento["texto"],
        config=types.EmbedContentConfig(
            # RETRIEVAL_DOCUMENT porque esto es un documento del corpus.
            # La pregunta del usuario se embebe con RETRIEVAL_QUERY. Son dos espacios
            # distintos que el modelo alinea a propósito: no se pueden mezclar.
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=DIMENSIONES,
        ),
    )
    vector = respuesta.embeddings[0].values
    fragmento["vector"] = [round(x, 6) for x in normalizar(vector)]
    print(f"  [{i}/{len(fragmentos)}] {fragmento['fuente']} — {fragmento['titulo']}")


# --- Paso 3: escribir el índice ------------------------------------------------------

indice = {
    "modelo": MODELO_EMBEDDINGS,
    "dimensiones": DIMENSIONES,
    "task_type_documentos": "RETRIEVAL_DOCUMENT",
    "fragmentos": fragmentos,
}

ARCHIVO_SALIDA.write_text(
    json.dumps(indice, ensure_ascii=False, indent=1), encoding="utf-8"
)

peso_kb = ARCHIVO_SALIDA.stat().st_size / 1024
print(f"\nListo: {ARCHIVO_SALIDA.name} ({peso_kb:.0f} KB, {len(fragmentos)} fragmentos)")
