import os
import queue
import threading
import json
import time

from core.video_pipeline import download_video, extract_audio, transcribe
from core.chunker import chunk_transcript
from ai.qa_generator_parallel import process_chunk

VIDEO_LIST_FILE = "videos.txt"

download_queue = queue.Queue()
transcribe_queue = queue.Queue()
llm_queue = queue.Queue()

MAX_DOWNLOAD_WORKERS = 2
MAX_LLM_WORKERS = 4


def load_urls():
    with open(VIDEO_LIST_FILE) as f:
        return [x.strip() for x in f if x.strip()]


def downloader_worker():
    while True:
        try:
            vid_idx, url = download_queue.get(timeout=5)
        except queue.Empty:
            return

        work_dir = os.path.join("work", f"video_{vid_idx}")
        os.makedirs(work_dir, exist_ok=True)

        print(f"⬇ Downloading Video {vid_idx}")

        video_path = download_video(url, work_dir)

        transcribe_queue.put((vid_idx, video_path, work_dir))

        download_queue.task_done()


def transcriber_worker():
    while True:
        try:
            vid_idx, video_path, work_dir = transcribe_queue.get(timeout=5)
        except queue.Empty:
            return

        print(f"🎤 Transcribing Video {vid_idx}")

        audio_path = extract_audio(video_path, work_dir)
        transcript = transcribe(audio_path)

        chunks = chunk_transcript(transcript)

        for i, ch in enumerate(chunks):
            llm_queue.put((vid_idx, i, ch["text"]))

        transcribe_queue.task_done()


def llm_worker():
    while True:
        try:
            vid_idx, chunk_id, text = llm_queue.get(timeout=5)
        except queue.Empty:
            return

        print(f"🧠 LLM Processing V{vid_idx} C{chunk_id}")

        result = process_chunk(chunk_id, {"text": text})

        out_dir = os.path.join("datasets", f"video_{vid_idx}")
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, f"chunk_{chunk_id}.json"), "w") as f:
            json.dump(result, f)

        llm_queue.task_done()


def main():
    urls = load_urls()

    for i, url in enumerate(urls):
        download_queue.put((i, url))

    start = time.time()

    threads = []

    # Start download workers
    for _ in range(MAX_DOWNLOAD_WORKERS):
        t = threading.Thread(target=downloader_worker)
        t.start()
        threads.append(t)

    # Start transcriber worker
    t = threading.Thread(target=transcriber_worker)
    t.start()
    threads.append(t)

    # Start LLM workers
    for _ in range(MAX_LLM_WORKERS):
        t = threading.Thread(target=llm_worker)
        t.start()
        threads.append(t)

    # Wait for all queues to finish
    download_queue.join()
    transcribe_queue.join()
    llm_queue.join()

    end = time.time()

    print("\n🚀 STREAMING PIPELINE COMPLETE")
    print("Total Time:", round(end - start, 2), "sec")


if __name__ == "__main__":
    main()