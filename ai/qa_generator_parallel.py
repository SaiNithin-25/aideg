import json
from runtime_bootstrap import ensure_local_site_packages

ensure_local_site_packages()

import requests
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pynvml 

# File Paths
INPUT_FILE = "data/chunks.json"
OUTPUT_FILE = "data/qa_dataset.json"
FAILED_LOG = "data/failed_chunks.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
MAX_RETRIES = 3
TIMEOUT = 120
NUM_WORKERS = 4 

# Thread-safe counter for active workers
active_workers = 0
counter_lock = threading.Lock()

# Initialize GPU monitoring
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    gpu_handle = None

def get_gpu_stats():
    if not gpu_handle:
        return "GPU Usage: N/A", "VRAM: N/A"
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        vram_used = round(info.used / (1024**3), 2)
        vram_total = round(info.total / (1024**3), 2)
        return f"GPU Usage: {util.gpu}%", f"VRAM: {vram_used}GB / {vram_total}GB"
    except:
        return "GPU Usage: Err", "VRAM: Err"

def call_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300}
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
{{"question": "...", "answer": "..."}}
]
}}
Return ONLY valid JSON.
"""

def process_chunk(index, chunk):
    global active_workers
    # Increment active workers when this specific thread starts working
    with counter_lock:
        active_workers += 1

    prompt = build_prompt(chunk["text"])
    result = {"status": "failed", "chunk_id": index, "content": chunk["text"]}

    try:
        for attempt in range(MAX_RETRIES):
            try:
                output = call_llm(prompt)
                match = re.search(r'{.*}', output, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    result = {
                        "status": "success",
                        "chunk_id": index,
                        "content": chunk["text"],
                        "dataset": parsed
                    }
                    break
            except Exception:
                time.sleep(2)
    finally:
        # Decrement active workers when this thread is finished
        with counter_lock:
            active_workers -= 1
    return result

def main(input_file=INPUT_FILE, output_file=OUTPUT_FILE, failed_log=FAILED_LOG):
    # Load data
    try:
        with open(input_file, "r") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    dataset = []
    failed = []
    total_chunks = len(chunks)
    start_total = time.time()

    print(f"🚀 Starting processing {total_chunks} chunks with {NUM_WORKERS} workers...\n")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_chunk, i, ch) for i, ch in enumerate(chunks)]

        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success":
                dataset.append(res)
            else:
                failed.append(res)

            # Update stats display
            gpu_usage, vram_usage = get_gpu_stats()
            loaded_count = len(dataset) + len(failed)
            
            # Print the real-time status line
            print(
                f"Status: Works loading... (Active: {active_workers}/{NUM_WORKERS}) | "
                f"Num of works loaded: {loaded_count}/{total_chunks} | "
                f"{gpu_usage} | {vram_usage}    ", 
                end="\r"
            )

    end_total = time.time()
    
    # Sort and Save
    dataset.sort(key=lambda x: x["chunk_id"])
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    with open(failed_log, "w") as f:
        json.dump(failed, f, indent=2)

    # Final Summary
    print("\n\n✅ Parallel dataset generation complete")
    print("Total Time:", round(end_total - start_total, 2), "sec")
    print("Success:", len(dataset))
    print("Failed:", len(failed))

if __name__ == "__main__":
    main()
