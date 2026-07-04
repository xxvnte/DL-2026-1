# metrics.py
import json
import re
from sentence_transformers import SentenceTransformer, util

model_emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with open("test_reference.jsonl", encoding="utf-8") as f:
    referencias = [json.loads(l)["cuerpo"] for l in f]

with open("dataset_sintetico.jsonl", encoding="utf-8") as f:
    generados = [json.loads(l) for l in f]

ref_emb = model_emb.encode(referencias, convert_to_tensor=True)

for g in generados:
    g_emb = model_emb.encode(g["texto"], convert_to_tensor=True)
    sim = util.cos_sim(g_emb, ref_emb).max().item()
    g["relevancia_semantica"] = round(sim, 3)


def distinct_n(textos, n=2):
    todos_ngrams, unicos = [], set()
    for t in textos:
        tokens = t.split()
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        todos_ngrams += ngrams
        unicos.update(ngrams)
    return len(unicos) / max(len(todos_ngrams), 1)


def dividir_oraciones(texto):
    oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())
    return [o for o in oraciones if len(o.split()) > 2]


def coherencia_intra_texto(texto):
    oraciones = dividir_oraciones(texto)
    if len(oraciones) < 2:
        return None
    embs = model_emb.encode(oraciones, convert_to_tensor=True)
    sims = []
    for i in range(len(oraciones) - 1):
        sims.append(util.cos_sim(embs[i], embs[i + 1]).item())
    return sum(sims) / len(sims)


for g in generados:
    g["coherencia"] = coherencia_intra_texto(g["texto"])


def evaluar_formato(texto, min_palabras=150, max_palabras=200, requiere_categoria=True):
    checks = {}
    n_palabras = len(texto.split())
    checks["longitud_ok"] = min_palabras <= n_palabras <= max_palabras
    checks["n_palabras"] = n_palabras
    if requiere_categoria:
        checks["tiene_categoria"] = bool(
            re.search(
                r"\b(nacional|local|internacional|economía|deportes|tecnología|política)\b",
                texto,
                re.IGNORECASE,
            )
        )
    checks["cumple_formato"] = checks["longitud_ok"] and checks.get(
        "tiene_categoria", True
    )
    return checks


for g in generados:
    fmt = evaluar_formato(g["texto"])
    g["formato"] = fmt

print(f"\n{'='*70}")
print(
    f"{'MODELO':<10} {'DISTINCT-2':<12} {'RELEVANCIA':<12} {'COHERENCIA':<12} {'FORMATO OK':<12}"
)
print(f"{'='*70}")

for familia in sorted(set(g["modelo"] for g in generados)):
    subset = [g for g in generados if g["modelo"] == familia]
    textos = [g["texto"] for g in subset]

    d2 = distinct_n(textos)
    rel_prom = sum(g["relevancia_semantica"] for g in subset) / len(subset)
    coh_vals = [g["coherencia"] for g in subset if g["coherencia"] is not None]
    coh_prom = sum(coh_vals) / len(coh_vals) if coh_vals else float("nan")
    formato_ok_pct = (
        sum(1 for g in subset if g["formato"]["cumple_formato"]) / len(subset) * 100
    )

    print(
        f"{familia:<10} {d2:<12.3f} {rel_prom:<12.3f} {coh_prom:<12.3f} {formato_ok_pct:<11.1f}%"
    )

with open("resultados_metricas.jsonl", "w", encoding="utf-8") as f:
    for g in generados:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")

print(f"\nDetalle completo guardado en resultados_metricas.jsonl")
