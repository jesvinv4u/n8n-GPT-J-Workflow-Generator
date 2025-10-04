# n8n Workflow Generator with GPT-J

This project allows you to **generate fully valid n8n workflows from natural language descriptions** using a fine-tuned GPT-J model. By providing a plain English description of the desired workflow, the model outputs a JSON structure compatible with n8n’s workflow format, including nodes, connections, and settings.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Workflow Logic](#workflow-logic)  
- [Notes & Limitations](#notes--limitations)  

---

## Overview

n8n is a workflow automation tool that allows users to automate tasks between apps and services. Each workflow is represented as JSON, containing:

- `nodes` – individual actions or triggers  
- `connections` – edges between nodes, defining data flow  
- `settings` – workflow-wide configuration  

This project uses **GPT-J**, a large language model, to generate such workflow JSON automatically from user prompts.

---

## Features

- Generate workflows for triggers, conditional logic, API calls, and AI integrations.  
- Support for complex workflows including multi-node sequences.  
- Few-shot learning with example workflows to guide generation.  
- Designed for GPT-J, capable of handling long and structured outputs.  

---

## Requirements

- **GPT-J model**: Fine-tuned GPT-J checkpoint trained on example n8n workflows.  
- **Hardware**: GPU with sufficient VRAM (at least 12–16GB recommended). CPU inference is possible but slow.  
- **Python >= 3.8**  
- **Libraries**: `transformers`, `torch`  

> ⚠️ **Important:** This project **cannot be trained or fine-tuned on GPT-2** or smaller models for production-level reliability. GPT-J’s capacity is required to reliably generate complex workflows.  

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd n8n-workflow-generator
Install dependencies:

bash
pip install torch transformers
Place the fine-tuned GPT-J model checkpoint in ./gptj-n8n.

Store example workflows in examples/workflows/<workflow_name>.json for few-shot learning.

Usage
Run the workflow generator script:

bash
python generate_workflow.py
Enter a natural language description when prompted:

pgsql
Enter your workflow description: Create a Telegram AI assistant that replies to voice messages
The script will output a JSON workflow compatible with n8n:

json
{
  "name": "Telegram AI Assistant",
  "nodes": [...],
  "connections": {...},
  "settings": {}
}
Import the JSON directly into n8n via Workflows → Import.

Workflow Logic
Input Collection: User provides a workflow description in natural language.

Prompt Construction:

GPT-J receives a “system prompt” describing the task.

Few-shot examples (stored in examples/workflows/) are prepended to show desired JSON patterns.

The user prompt is appended.

Generation: GPT-J produces workflow JSON including:

Nodes (type, parameters, id, position)

Connections (data flow between nodes)

Settings (active, executionOrder)

Post-Processing:

JSON is validated.

Node IDs and webhook IDs can be automatically replaced or updated.

Output: JSON can be imported directly into n8n, making the workflow executable.

Notes & Limitations
GPT-J is required due to context length and model capacity. Smaller models like GPT-2 are insufficient for reliable multi-node workflow generation.

Generated JSON may occasionally need minor validation or correction.

Extremely large workflows may hit GPT-J’s context window limit; consider splitting complex workflows into smaller prompts.

This project is designed only for training with required resources (GPU, memory, and fine-tuned dataset). Training with inadequate resources may fail or produce low-quality outputs.

License
MIT License – free to use and modify, but ensure GPT-J licensing compliance.











