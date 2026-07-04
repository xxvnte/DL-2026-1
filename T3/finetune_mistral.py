from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import sys

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
OUT_DIR = "mistral_noticias_gguf"

max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=4,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files="train.jsonl", split="train")


def format_example(ex):
    text = f"### Instrucción:\n{ex['instruction']}\n\n### Noticia:\n{ex['output']}"
    return {"text": text}


dataset = dataset.map(format_example)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs_mistral",
        optim="adamw_8bit",
        report_to="none",
    ),
)

trainer.train()

model.save_pretrained_gguf(OUT_DIR, tokenizer, quantization_method="q4_k_m")
print(f"GGUF guardado en {OUT_DIR}")
