import json
from runtime_bootstrap import ensure_local_site_packages

ensure_local_site_packages()

import requests
import time
import re

INPUT_FILE = "data/chunks.json"
OUTPUT_FILE = "data/qa_dataset.json"
FAILED_LOG = "data/failed_chunks.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

MAX_RETRIES = 3
TIMEOUT = 120


def load_chunks():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)


def call_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300
            }
        },
        timeout=TIMEOUT
    )
    return response.json()["response"]


def build_prompt(text):
    return f"""
You are a dataset generation AI.

From the lecture content below:

{text}

Generate STRICT JSON only:

{{
  "summary": "...",
  "qa_pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}

Return ONLY valid JSON.
"""


def extract_json(text):
    match = re.search(r'{.*}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def process_chunk(chunk):
    prompt = build_prompt(chunk["text"])

    for attempt in range(MAX_RETRIES):
        try:
            output = call_llm(prompt)
            json_text = extract_json(output)

            if json_text is None:
                raise ValueError("No JSON found")

            parsed = json.loads(json_text)
            return parsed

        except Exception as e:
            print(f"Retry {attempt+1} failed:", str(e))
            time.sleep(2)

    return None


def main():
    chunks = load_chunks()
    dataset = []
    failed = []

    for i, ch in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        start = time.time()

        result = process_chunk(ch)
        end = time.time()

        if result:
            dataset.append({
                "chunk_id": i,
                "content": ch["text"],
                "dataset": result,
                "time": round(end - start, 2)
            })
        else:
            failed.append({
                "chunk_id": i,
                "content": ch["text"]
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    with open(FAILED_LOG, "w") as f:
        json.dump(failed, f, indent=2)

    print("\n✅ Dataset generation finished")
    print(f"Successful: {len(dataset)}")
    print(f"Failed: {len(failed)}")


if __name__ == "__main__":
    main()
