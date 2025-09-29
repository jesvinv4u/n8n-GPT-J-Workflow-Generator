# n8n_gptj_finetune.py

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
data_path = r"D:/Project/n8n/n8n_docs.jsonl"  # your dataset path
dataset = load_dataset("json", data_files=data_path, split="train")

# -----------------------------
# 2️⃣ Load GPT-J tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token  # avoid padding issues

# -----------------------------
# 3️⃣ Tokenize dataset
# -----------------------------
def tokenize_fn(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,  # GPT-J can handle long sequences
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# -----------------------------
# 4️⃣ Load GPT-J model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# -----------------------------
# 5️⃣ Setup training arguments
# -----------------------------
output_dir = r"D:/Project/n8n/n8n-gpt-j"  # folder to save fine-tuned model

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # reduce if GPU memory is limited
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=50,
    save_strategy="epoch",
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # mixed precision if GPU is available
    report_to=None  # disable wandb or other loggers
)

# -----------------------------
# 6️⃣ Initialize Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -----------------------------
# 7️⃣ Train model
# -----------------------------
print("Starting training...")
trainer.train()

# -----------------------------
# 8️⃣ Save fine-tuned model and tokenizer
# -----------------------------
print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete! Your fine-tuned GPT-J model is ready.")
