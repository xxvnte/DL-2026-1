import argparse
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_ORDER = ["llama", "phi", "mistral"]
MODEL_LABELS = {"llama": "LLaMA", "phi": "Phi", "mistral": "Mistral"}
MODEL_COLORS = {"llama": "#4C72B0", "phi": "#DD8452", "mistral": "#55A868"}


def _boxplot_compat(ax, data, labels, **kwargs):
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)


def load_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataframe(records):
    rows = []
    for r in records:
        formato = r.get("formato", {}) or {}
        rows.append(
            {
                "modelo": r["modelo"],
                "prompt_id": r["prompt_id"],
                "texto": r["texto"],
                "coherencia": r["coherencia"],
                "relevancia_semantica": r["relevancia_semantica"],
                "n_palabras": formato.get("n_palabras"),
                "longitud_ok": bool(formato.get("longitud_ok")),
                "tiene_categoria": bool(formato.get("tiene_categoria")),
                "cumple_formato": bool(formato.get("cumple_formato")),
            }
        )
    df = pd.DataFrame(rows)
    present = [m for m in MODEL_ORDER if m in df["modelo"].unique()]
    otros = [m for m in df["modelo"].unique() if m not in present]
    df["modelo"] = pd.Categorical(
        df["modelo"], categories=present + otros, ordered=True
    )
    return df.sort_values(["modelo", "prompt_id"]).reset_index(drop=True)


def _tokenize(text):
    return re.findall(r"[a-záéíóúñü]+", text.lower())


def _distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def _tfidf_matrix(texts):
    docs_tokens = [_tokenize(t) for t in texts]
    vocab = sorted(set(tok for toks in docs_tokens for tok in toks))
    idx = {w: i for i, w in enumerate(vocab)}
    n_docs, n_vocab = len(texts), len(vocab)

    tf = np.zeros((n_docs, n_vocab))
    for i, toks in enumerate(docs_tokens):
        for tok in toks:
            tf[i, idx[tok]] += 1
        if toks:
            tf[i] /= len(toks)

    df_counts = (tf > 0).sum(axis=0)
    idf = np.log((n_docs + 1) / (df_counts + 1)) + 1
    tfidf = tf * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return tfidf / norms


def _cosine_sim(tfidf_norm):
    return tfidf_norm @ tfidf_norm.T


def compute_diversity(df):
    rows = []
    for modelo, sub in df.groupby("modelo", observed=True):
        textos = sub["texto"].tolist()
        d1 = float(np.mean([_distinct_n(_tokenize(t), 1) for t in textos]))
        d2 = float(np.mean([_distinct_n(_tokenize(t), 2) for t in textos]))

        if len(textos) > 1:
            tfidf = _tfidf_matrix(textos)
            sim = _cosine_sim(tfidf)
            pares = list(combinations(range(len(textos)), 2))
            sim_prom = float(np.mean([sim[i, j] for i, j in pares]))
        else:
            sim_prom = 0.0

        rows.append(
            {
                "modelo": modelo,
                "distinct_1": d1,
                "distinct_2": d2,
                "similitud_interexemplos": sim_prom,
                "diversidad_interexemplos": 1 - sim_prom,
            }
        )
    return pd.DataFrame(rows)


def compute_summary(df, diversity_df):
    agg = (
        df.groupby("modelo", observed=True)
        .agg(
            coherencia_media=("coherencia", "mean"),
            coherencia_std=("coherencia", "std"),
            relevancia_media=("relevancia_semantica", "mean"),
            relevancia_std=("relevancia_semantica", "std"),
            n_palabras_media=("n_palabras", "mean"),
            n_palabras_std=("n_palabras", "std"),
            pct_cumple_formato=("cumple_formato", "mean"),
            pct_tiene_categoria=("tiene_categoria", "mean"),
            pct_longitud_ok=("longitud_ok", "mean"),
            n=("prompt_id", "count"),
        )
        .reset_index()
    )

    for col in ["pct_cumple_formato", "pct_tiene_categoria", "pct_longitud_ok"]:
        agg[col] = agg[col] * 100

    return agg.merge(diversity_df, on="modelo").reset_index(drop=True)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_coherencia_relevancia_box(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    metrics = [
        ("coherencia", "Coherencia del texto generado"),
        ("relevancia_semantica", "Relevancia semántica respecto al tópico"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        cats = list(df["modelo"].cat.categories)
        data = [df.loc[df["modelo"] == m, metric].dropna().values for m in cats]
        bp = _boxplot_compat(
            ax, data, [MODEL_LABELS[m] for m in cats], patch_artist=True, showmeans=True
        )
        for patch, m in zip(bp["boxes"], cats):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Distribución de coherencia y relevancia semántica por modelo", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "01_coherencia_relevancia_boxplot.png"), dpi=150
    )
    plt.close(fig)


def plot_cumplimiento_formato(summary, output_dir):
    modelos = summary["modelo"].tolist()
    x = np.arange(len(modelos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width,
        summary["pct_longitud_ok"],
        width,
        label="Longitud OK",
        color="#8172B2",
    )
    ax.bar(
        x,
        summary["pct_tiene_categoria"],
        width,
        label="Tiene categoría",
        color="#C44E52",
    )
    ax.bar(
        x + width,
        summary["pct_cumple_formato"],
        width,
        label="Cumple formato completo",
        color="#55A868",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in modelos])
    ax.set_ylabel("% de ejemplos")
    ax.set_ylim(0, 105)
    ax.set_title("Cumplimiento del formato solicitado por modelo")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_cumplimiento_formato.png"), dpi=150)
    plt.close(fig)


def plot_diversidad(summary, output_dir):
    modelos = summary["modelo"].tolist()
    x = np.arange(len(modelos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        summary["distinct_2"],
        width,
        label="Distinct-2 (riqueza léxica intra-texto)",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        summary["diversidad_interexemplos"],
        width,
        label="1 - similitud TF-IDF entre ejemplos",
        color="#DD8452",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in modelos])
    ax.set_ylabel("score (0-1)")
    ax.set_ylim(0, 1)
    ax.set_title("Diversidad de los ejemplos generados por modelo")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_diversidad.png"), dpi=150)
    plt.close(fig)


def plot_longitud(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(df["modelo"].cat.categories)
    data = [df.loc[df["modelo"] == m, "n_palabras"].dropna().values for m in cats]
    bp = _boxplot_compat(
        ax, data, [MODEL_LABELS[m] for m in cats], patch_artist=True, showmeans=True
    )
    for patch, m in zip(bp["boxes"], cats):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.6)
    ax.set_ylabel("n° de palabras")
    ax.set_title("Longitud de los textos generados por modelo")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_longitud_textos.png"), dpi=150)
    plt.close(fig)


def plot_radar(summary, output_dir):
    criterios = [
        "Coherencia",
        "Relevancia\nsemántica",
        "Diversidad",
        "Cumplimiento\nde formato",
    ]

    rel = summary["relevancia_media"]
    rel_min, rel_max = rel.min(), rel.max()
    rel_norm = (
        (rel - rel_min) / (rel_max - rel_min) if rel_max > rel_min else rel * 0 + 0.5
    )
    coh_norm = summary["coherencia_media"].clip(0, 1)

    angles = np.linspace(0, 2 * np.pi, len(criterios), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, m in enumerate(summary["modelo"]):
        vals = [
            coh_norm.iloc[i],
            rel_norm.iloc[i],
            summary["diversidad_interexemplos"].iloc[i],
            summary["pct_cumple_formato"].iloc[i] / 100,
        ]
        vals += vals[:1]
        ax.plot(angles, vals, label=MODEL_LABELS[m], color=MODEL_COLORS[m], linewidth=2)
        ax.fill(angles, vals, color=MODEL_COLORS[m], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criterios)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Comparación general LLaMA vs Phi vs Mistral\n(escala normalizada 0-1)", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_radar_comparativo.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="resultados_metricas.jsonl")
    parser.add_argument("--output-dir", default="graficas")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    records = load_data(args.input)
    df = build_dataframe(records)
    diversity_df = compute_diversity(df)
    summary = compute_summary(df, diversity_df)

    plot_coherencia_relevancia_box(df, args.output_dir)
    plot_cumplimiento_formato(summary, args.output_dir)
    plot_diversidad(summary, args.output_dir)
    plot_longitud(df, args.output_dir)
    plot_radar(summary, args.output_dir)

    summary_out = summary.copy()
    summary_out["modelo"] = summary_out["modelo"].map(MODEL_LABELS)
    summary_out.to_csv(
        os.path.join(args.output_dir, "resumen_metricas.csv"), index=False
    )

    print(summary_out.round(3).to_string(index=False))
    print(f"\nGráficas guardadas en: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
