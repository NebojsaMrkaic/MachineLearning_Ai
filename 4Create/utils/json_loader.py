import json

def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)
