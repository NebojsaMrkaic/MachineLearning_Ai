import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 🔐 Učitaj API ključ (uvijek se izvršava)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY nije pronađen u .env fajlu.")

client = OpenAI(api_key=api_key)


def query_openai(prompt, document_text):
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": f"{prompt['user']}\n\nDocument:\n{document_text}"}
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
        print("❌ Odgovor nije validan JSON.")
        return {"error": "Invalid JSON response", "raw": response_text}


# 🧪 Test kod koji se pokreće SAMO ako se fajl pokrene direktno
if __name__ == "__main__":
    print("🧪 OpenAI Client - Test Mode")

    # Testni prompt
    test_prompt = {
        "model": "gpt-4o-mini",
        "system": "You are a helpful assistant.",
        "user": "Say hello and tell me what time it is."
    }

    try:
        response = query_openai(test_prompt, "Test document")
        print(f"✅ API poziv uspešan!")
        print(f"📄 Odgovor: {response}")

        parsed = parse_response_to_json(response)
        print(f"🔧 Parsiran JSON: {json.dumps(parsed, indent=2, ensure_ascii=False)}")

    except Exception as e:
        print(f"❌ Greška pri testu: {e}")