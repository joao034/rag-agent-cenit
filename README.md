# Agente RAG Cénit

Un agente conversacional con arquitectura **RAG (Retrieval-Augmented Generation)**
que responde preguntas apoyándose en una base de conocimiento propia y en
herramientas que consultan datos en tiempo real, en lugar de responder solo
con lo que el modelo "recuerda".

## Qué hace

El agente recibe una pregunta en lenguaje natural y decide, por sí mismo, qué
herramienta necesita para responderla:

- **Buscar en la base de conocimiento** — cuando la pregunta es sobre reglas,
  requisitos o información documentada (RAG puro: embeddings + búsqueda por
  similitud).
- **Calcular un costo** — cuando la pregunta implica un cálculo con datos
  estructurados (precios, descuentos, impuestos).
- **Consultar disponibilidad** — cuando la pregunta necesita un dato que
  cambia en tiempo real, simulando una llamada a un servicio externo.

Puede encadenar varias herramientas en una sola pregunta (por ejemplo,
"¿cuánto me sale este curso con esta beca y todavía hay lugar?" dispara dos
herramientas), y siempre expone la **trayectoria** completa: qué herramientas
usó, en qué orden y con qué argumentos, para que la respuesta sea auditable.

## Cómo funciona el RAG

1. **Indexado (offline, una sola vez):** los documentos de la base de
   conocimiento se dividen en fragmentos y se convierten en vectores con el
   modelo de embeddings de Gemini. El resultado se guarda en `index.json`.
2. **Recuperación (en cada pregunta):** la pregunta del usuario se convierte
   en un vector con el mismo modelo de embeddings, y se compara por
   similitud coseno contra todos los fragmentos del índice.
3. **Umbral de confianza:** si el fragmento más parecido no supera un umbral
   mínimo, el agente responde que no tiene esa información documentada, en
   lugar de inventar una respuesta con fragmentos poco relacionados.
4. **Generación aumentada:** los 3 fragmentos más relevantes (con su fuente y
   su score de similitud) se le entregan al modelo de lenguaje como contexto,
   para que redacte la respuesta final citando esa información.

## Stack técnico

| Componente               | Detalle                                                             |
|---------------------------|-----------------------------------------------------------------------|
| Modelo de lenguaje (LLM)  | `openai/gpt-oss-120b` servido vía **Groq** (function-calling)          |
| Modelo de embeddings      | `gemini-embedding-001` de **Google Gemini** (768 dimensiones)          |
| Similitud                 | Coseno, con `numpy`                                                     |
| Índice                    | Vectores precomputados en `index.json`, cargados en memoria             |
| API                        | **FastAPI**, con página de prueba servida en `/index`                          |
| Orquestación del agente    | Loop propio con límite de vueltas y manejo de errores por herramienta      |

## Estructura del proyecto

```
agente-rag-cenit/
├── main.py           # FastAPI: la app y los 3 endpoints (/, /health, /preguntar)
├── agent.py           # El loop del agente: llama al modelo, ejecuta herramientas, repite
├── tools.py            # Las herramientas (incluida la búsqueda RAG) + su esquema
├── data.py              # Datos de negocio (precios, becas, cupos)
├── config.py             # Constantes, clientes (Groq/Gemini) y carga del índice
├── static/index.html      # Página mínima para probar el agente desde el navegador
├── indice.json             # Índice de embeddings de la base de conocimiento
├── requirements.txt
└── .env.example
```

## Cómo correrlo

```bash
cp .env.example .env   # y llena las llaves
pip install -r requirements.txt

export GROQ_API_KEY=...
export GEMINI_API_KEY=...
uvicorn main:app --reload
```

Abre `http://localhost:8000` para probarlo desde el navegador, o llama
directamente al endpoint:

```bash
curl -X POST http://localhost:8000/preguntar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿cuánto me sale el curso CNT-204 en línea?"}'
```

La respuesta incluye tanto el texto final como la trayectoria de
herramientas usadas, para poder inspeccionar cómo llegó a esa respuesta.

