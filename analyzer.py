import os
import re
import json
from groq import Groq

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY no configurada")
        _client = Groq(api_key=api_key)
    return _client


def _call(prompt: str, max_tokens: int = 2000) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def analyze_publication(my_listing: dict, competitors: list) -> dict:
    competitors_text = ""
    for i, c in enumerate(competitors[:15], 1):
        price = f"${c.get('price'):,}" if c.get("price") else "N/D"
        rating = c.get("rating") or "sin calificación"
        reviews = c.get("reviews") or 0
        shipping = "envío gratis" if c.get("free_shipping") else "sin envío gratis"
        competitors_text += (
            f"{i}. {c.get('title','?')} | {price} | ★{rating} ({reviews} reseñas) | {shipping}\n"
        )

    attrs_text = json.dumps(my_listing.get("attributes", {}), ensure_ascii=False, indent=2)

    prompt = f"""Sos un experto en optimización de publicaciones de MercadoLibre Argentina.
Analizá mi publicación y comparala con los competidores para darme recomendaciones concretas y accionables.

## MI PUBLICACIÓN
- **Título**: {my_listing.get("title", "N/D")}
- **Precio**: ${my_listing.get("price", "N/D"):,} ARS
- **Calificación**: {my_listing.get("rating") or "Sin calificaciones aún"}
- **Reseñas**: {my_listing.get("reviews_count", 0)}
- **Fotos**: {my_listing.get("images_count", 0)} imágenes
- **Video**: {"Sí" if my_listing.get("has_video") else "No"}
- **Envío gratis**: {"Sí" if my_listing.get("free_shipping") else "No"}
- **Marca**: {my_listing.get("brand") or "Genérica"}
- **Reputación del vendedor**: {my_listing.get("seller_reputation") or "N/D"}
- **Atributos**: {attrs_text}
- **Descripción (primeros 500 chars)**: {(my_listing.get("description") or "")[:500]}

## TOP COMPETIDORES (búsqueda orgánica)
{competitors_text}

## TAREA
Analizá en profundidad y devolvé un JSON con esta estructura exacta (sin texto extra, solo JSON válido):
{{
  "score": <número 1-10 que representa la calidad actual de la publicación>,
  "score_reason": "<1 oración explicando el puntaje>",
  "summary": "<resumen ejecutivo de 2-3 oraciones del estado actual vs competidores>",
  "price_position": "<premium|competitivo|económico|muy económico> con una breve explicación",
  "proposed_title": "<título nuevo EXACTO que deberías usar, optimizado para ML: máx 60 caracteres, con keywords de alto volumen, sin repetir palabras>",
  "recommendations": [
    {{
      "priority": "<alta|media|baja>",
      "category": "<título|precio|fotos|video|descripción|envío|atributos|palabras_clave|oferta>",
      "issue": "<qué está mal o qué falta>",
      "action": "<qué hacer exactamente>",
      "proposed_value": "<el texto/valor EXACTO que debería ir: el título nuevo, el precio sugerido, el texto de descripción, etc. Siempre dar el reemplazo listo para copiar y pegar>",
      "impact": "<cuál es el impacto esperado en ventas>"
    }}
  ]
}}

IMPORTANTE: En "proposed_value" siempre poné el texto listo para usar, no instrucciones genéricas.
Para título: el título completo optimizado.
Para descripción: el párrafo o sección mejorada.
Para precio: el número exacto sugerido.
Ordená las recomendaciones de mayor a menor prioridad."""

    return json.loads(_call(prompt, max_tokens=2000))


def analyze_opportunities(vertical_data: list, selected_vertical: str) -> dict:
    data_text = json.dumps(vertical_data[:50], ensure_ascii=False, indent=2)

    prompt = f"""Sos un experto en importación y comercio electrónico, especializado en MercadoLibre Argentina.
Tenés datos del vertical "{selected_vertical}" de MercadoLibre. Analizalos para encontrar oportunidades de nicho
para importar productos desde China y vender en Argentina.

## DATOS DEL VERTICAL
{data_text}

## TAREA
Identificá los mejores nichos dentro de este vertical para importar y vender. Considerá:
- Demanda vs competencia (nichos con demanda pero pocos vendedores establecidos)
- Margen potencial (productos que se pueden importar a bajo costo)
- Tendencias del mercado
- Barreras de entrada bajas

Devolvé un JSON con esta estructura exacta (sin texto extra, solo JSON válido):
{{
  "vertical": "{selected_vertical}",
  "market_summary": "<resumen del mercado en 2-3 oraciones>",
  "opportunities": [
    {{
      "product": "<nombre del producto/nicho>",
      "score": <1-10 de atractivo para importar>,
      "demand_level": "<alta|media|baja>",
      "competition_level": "<alta|media|baja>",
      "estimated_import_cost": "<rango en USD>",
      "estimated_selling_price": "<rango en ARS>",
      "why": "<por qué es una buena oportunidad>",
      "key_differentiators": ["<diferenciador 1>", "<diferenciador 2>"],
      "search_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>"]
    }}
  ],
  "top_recommendation": "<el mejor producto específico para importar ahora y por qué>"
}}

Ordená las oportunidades de mayor a menor score."""

    return json.loads(_call(prompt, max_tokens=2500))
