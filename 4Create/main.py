import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
import time

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY has not been found within .env file.")

client = OpenAI(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def query_openai(prompt, document_text):
    messages = [
        {"role": "system", "content": prompt.get("system", "")},
        {"role": "user", "content": f"{prompt.get('user', '')}\n\nDocument:\n{document_text}"}
    ]
    response = client.chat.completions.create(
        model=prompt.get("model", "gpt-4o-mini"),
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content

def parse_response_to_json(response_text):
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            return json.loads(match.group(0))
        return {"error": "Invalid JSON response", "raw": response_text}

def run_agent(prompt_path, pdf_path, output_path):
    print(f"\n▶️ Running an agent prompt: {prompt_path}")
    start_time = time.time()

    prompt = load_prompt(prompt_path)
    document_text = extract_text_from_pdf(pdf_path)
    response_text = query_openai(prompt, document_text)
    parsed_response = parse_response_to_json(response_text)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_response, f, indent=2, ensure_ascii=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ Agent finished. Results are saved at: {output_path}")
    print(f"⏱️ Duration: {duration:.2f} seconds")

if __name__ == "__main__":
        run_agent("agent_v1.json", "test_doc.pdf", "output_v1.json")
        run_agent("agent_v2.json", "test_doc.pdf", "output_v2.json")

