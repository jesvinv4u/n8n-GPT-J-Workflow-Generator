# n8n-GPT-J-Workflow-Generator
This project fine-tunes GPT-J on the official n8n documentation and uses it to generate and deploy n8n workflows automatically.

Overview

Fine-tuned GPT-J
A 6-billion-parameter language model trained on n8n documentation so it can answer questions and draft workflow steps.

FastAPI Model Server
Serves the trained GPT-J model via REST (http://localhost:8000/generate).

Node.js App (app.js)
Accepts user input, queries the GPT-J server, and creates a new workflow inside a local n8n instance via the n8n REST API.

🗂 Project Structure
.
├─ app.js                 # Node script that generates & posts workflows to n8n
├─ server.py              # FastAPI server hosting the fine-tuned GPT-J model
├─ train_gptj_n8n.py      # Fine-tune script for GPT-J
├─ n8n_docs.jsonl         # Cleaned n8n docs for training
└─ README.md

🔧 Requirements

Python 3.9+
Packages: transformers, datasets, accelerate, fastapi, uvicorn

Node.js 18+
Package: axios

n8n running locally (http://localhost:5678)

⚡ Setup & Usage
1️⃣ Prepare Training Data

Convert docs to JSONL (if not already done):

python prepare_jsonl.py

2️⃣ Fine-Tune GPT-J
pip install transformers datasets accelerate
python train_gptj_n8n.py


train_gptj_n8n.py (key parts):

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

model_name = "EleutherAI/gpt-j-6B"
data_path  = "n8n_docs.jsonl"

dataset = load_dataset("json", data_files=data_path, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

args = TrainingArguments(
    output_dir="./n8n-gptj",
    per_device_train_batch_size=1,   # GPT-J is large—start small
    num_train_epochs=2,
    gradient_accumulation_steps=8,
    fp16=True,
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
trainer.save_model("./n8n-gptj")
tokenizer.save_pretrained("./n8n-gptj")


💡 GPT-J is big—training is GPU-heavy. For CPU training you’ll need a smaller batch size or to use parameter-efficient fine-tuning (LoRA, PEFT).

3️⃣ Serve the Model
pip install "uvicorn[standard]" fastapi
python -m uvicorn server:app --host 0.0.0.0 --port 8000


The endpoint POST /generate accepts:

{ "prompt": "How do I create a workflow in n8n?" }

4️⃣ Start n8n

Run n8n locally and create an API key:
Settings → API → Create Key

Set it in your environment:

set N8N_API_KEY=<your_key_here>      # Windows
export N8N_API_KEY=<your_key_here>   # Linux/Mac

5️⃣ Run the Node App

Install dependencies and start:

npm install
node app.js


app.js will:

Send your prompt to the GPT-J FastAPI server.

Receive generated workflow text.

POST a valid workflow payload to the n8n REST API.

🛠 How It Works

app.js

Sends user prompts to the GPT-J server.

Receives text describing a workflow.

Posts a valid workflow (nodes, connections, positions, settings) to POST /api/v1/workflows with the X-N8N-API-KEY header.

server.py

Loads the fine-tuned GPT-J model and exposes a /generate endpoint.

n8n REST API

Stores the workflow so you can view/activate it in the n8n UI.
