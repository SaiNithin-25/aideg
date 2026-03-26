import json
import requests
import time
import os
import subprocess

INPUT_FILE = "data/chunks.json"
OUTPUT_FILE = "data/qa_dataset.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

def load_chunks():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)

def ask_llm(prompt):
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
        }
    )
    return response.json()["response"]

def build_prompt(chunk_text):
    return f"""
You are an expert dataset generator AI.

From the lecture content below:

{chunk_text}

Generate dataset in STRICT JSON format:

{{
 "summary": "...",
 "qa_pairs": [
   {{"question": "...", "answer": "..."}},
   {{"question": "...", "answer": "..."}},
   {{"question": "...", "answer": "..."}}
 ]
}}

Return ONLY JSON.
"""

def get_gpu_usage():
    try:
        # Uses nvidia-smi to query GPU utilization
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        usage = result.stdout.strip().split(",")
        return {
            "gpu_utilization_percent": int(usage[0]),
            "gpu_memory_used_MB": int(usage[1])
        }
    except Exception:
        return {"gpu_utilization_percent": None, "gpu_memory_used_MB": None}

def main():
    chunks = load_chunks()
    dataset = []
    times = []

    for i, ch in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")

        prompt = build_prompt(ch["text"])

        start = time.time()
        result = ask_llm(prompt)
        end = time.time()

        gen_time = round(end - start, 2)
        times.append(gen_time)

        dataset.append({
            "chunk_id": i,
            "content": ch["text"],
            "qa_data": result,
            "generation_time": gen_time
        })

        print(f"Time: {gen_time} sec\n")

    # Save dataset
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    # Compute stats
    avg_time = round(sum(times) / len(times), 2)
    gpu_stats = get_gpu_usage()
    file_size_MB = round(os.path.getsize(OUTPUT_FILE) / (1024 * 1024), 2)

    print("Dataset generation complete ✅")
    print(f"Average generation time per chunk: {avg_time} sec")
    print(f"GPU usage: {gpu_stats}")
    print(f"Dataset file size: {file_size_MB} MB")

if __name__ == "__main__":
    main()