import re, json

def parse_news_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n(?=# )", content)
    examples = []
    for block in blocks:
        block = block.strip()
        if not block.startswith("# "):
            continue
        lines = block.split("\n")
        titulo = lines[0].replace("# ", "").strip()
        cuerpo = "\n".join(lines[1:])
        cuerpo = re.sub(r"```", "", cuerpo)
        cuerpo = re.sub(r"\*\*(.*?)\*\*", r"\1", cuerpo)
        cuerpo = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cuerpo)
        cuerpo = re.sub(r"#+\s*", "", cuerpo)
        cuerpo = re.sub(r"\n{2,}", "\n\n", cuerpo).strip()
        examples.append({"titulo": titulo, "cuerpo": cuerpo})
    return examples


data = parse_news_file("noticias_reales.txt")
print(f"{len(data)} noticias extraídas")

train, test = data[:5], data[5:]

with open("train.jsonl", "w", encoding="utf-8") as f:
    for ex in train:
        prompt = f"Escribe una noticia periodística en español sobre: {ex['titulo']}"
        f.write(
            json.dumps(
                {"instruction": prompt, "output": ex["cuerpo"]}, ensure_ascii=False
            )
            + "\n"
        )

with open("test_reference.jsonl", "w", encoding="utf-8") as f:
    for ex in test:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("train.jsonl y test_reference.jsonl generados")
