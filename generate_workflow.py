from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load GPT-J 6B model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained("./n8n-gpt2")
model = AutoModelForCausalLM.from_pretrained("./n8n-gpt2")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Ask user for workflow description
user_prompt = input("Enter your workflow description: ")

# Prepare input for GPT-J
input_text = f"PROMPT: {user_prompt} WORKFLOW: Return fully valid n8n workflow JSON."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate workflow JSON
outputs = model.generate(
    input_ids,
    max_length=1000,  # Was 400, too short for multi-node workflows
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

workflow_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated n8n workflow:\n")
print(workflow_text)
