from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
generator = pipeline("text-generation", model="./n8n-gpt2")

class Query(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(q: Query):
    result = generator(q.prompt, max_length=200, num_return_sequences=1)
    return {"text": result[0]["generated_text"]}
