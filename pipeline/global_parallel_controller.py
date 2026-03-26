import os
import json
import time
import queue
import threading

from core.video_pipeline import download_video, extract_audio, transcribe
from core.chunker import chunk_transcript
from ai.qa_generator_parallel import process_chunk   # reuse chunk logic

VIDEO_LIST_FILE = "videos.txt"

MAX_GLOBAL_WORKERS = 4   # ⭐ total LLM concurrency limit

task_queue = queue.Queue()
results_lock = threading.Lock()


def load_urls():
    with open(VIDEO_LIST_FILE) as f:
        return [x.strip() for x in f if x.strip()]


def producer():
    urls = load_urls()

    for vid_idx, url in enumerate(urls):
        print(f"\n🎬 Preparing Video {vid_idx}")

        work_dir = os.path.join("work", f"video_{vid_idx}")
        os.makedirs(work_dir, exist_ok=True)

        video_path = download_video(url, work_dir)
        audio_path = extract_audio(video_path, work_dir)
        transcript = transcribe(audio_path)

        chunks = chunk_transcript(transcript)

        for chunk_idx, ch in enumerate(chunks):
            task_queue.put({
                "video_id": vid_idx,
                "chunk_id": chunk_idx,
                "text": ch["text"],
                "work_dir": work_dir
            })

    print("\n✅ All tasks queued")


def worker():
    while True:
        try:
            task = task_queue.get(timeout=5)
        except queue.Empty:
            return

        video_id = task["video_id"]
        chunk_id = task["chunk_id"]

        result = process_chunk(chunk_id, {"text": task["text"]})

        out_dir = os.path.join("datasets", f"video_{video_id}")
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, f"chunk_{chunk_id}.json")

        with results_lock:
            with open(out_file, "w") as f:
                json.dump(result, f)

        task_queue.task_done()
        print(f"🧠 Processing Video {video_id} Chunk {chunk_id}")


def main():
    start = time.time()

    prod_thread = threading.Thread(target=producer)
    prod_thread.start()

    workers = []
    for _ in range(MAX_GLOBAL_WORKERS):
        t = threading.Thread(target=worker)
        t.start()
        workers.append(t)

    prod_thread.join()
    task_queue.join()

    for w in workers:
        w.join()

    end = time.time()
    print("\n🚀 GLOBAL PIPELINE COMPLETE")
    print("Total Time:", round(end - start, 2), "sec")


if __name__ == "__main__":
    main()