import ollama, json

MODELOS = {
    "llama": "llama3.1:8b",
    "phi": "phi3-noticias",
    "mistral": "mistral:7b",
}

PROMPTS = [
    "Escribe una noticia periodística simulada en español sobre un accidente de tránsito en Santiago. Formato: titular + cuerpo de 150-200 palabras. Incluye una categoría: [Nacional/Local].",
    "Escribe una noticia periodística simulada en español sobre un descubrimiento científico reciente. Formato: titular + cuerpo de 150-200 palabras. Incluye una categoría: [Ciencia/Tecnología].",
    "Escribe una noticia periodística simulada en español sobre un evento deportivo importante. Formato: titular + cuerpo de 150-200 palabras. Incluye una categoría: [Deportes].",
    "Escribe una noticia periodística simulada en español sobre una nueva medida económica del gobierno de Chile. Formato: titular + cuerpo de 150-200 palabras. Incluye una categoría: [Economía/Nacional].",
    "Escribe una noticia periodística simulada en español sobre el lanzamiento de un nuevo producto tecnológico de una empresa chilena. Formato: titular + cuerpo de 150-200 palabras. Incluye una categoría: [Tecnología].",
    "Escribe una noticia periodística simulada en español sobre un hecho de la vida política local (municipal o regional). Formato: titular + cuerpo de 150-200 palabras. Incluye una categoría: [Política/Local].",
]

resultados = []
for familia, modelo in MODELOS.items():
    for i, prompt in enumerate(PROMPTS):
        resp = ollama.generate(
            model=modelo, prompt=prompt, options={"temperature": 0.8, "top_p": 0.9}
        )
        resultados.append(
            {"modelo": familia, "prompt_id": i, "texto": resp["response"]}
        )

with open("dataset_sintetico.jsonl", "w", encoding="utf-8") as f:
    for r in resultados:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
