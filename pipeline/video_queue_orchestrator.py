import json
import os
import time

from ai.qa_generator_parallel import main as generate_dataset
from core.chunker import chunk_transcript
from core.video_pipeline import (
    download_video,
    extract_audio,
    save_transcript,
    transcribe,
)

VIDEO_LIST_FILE = "videos.txt"
BASE_OUTPUT_DIR = "datasets"


def load_video_list():
    with open(VIDEO_LIST_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_video_folder(index):
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    folder = os.path.join(BASE_OUTPUT_DIR, f"video_{index}")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_status(index):
    with open("pipeline_status.json", "w") as f:
        json.dump({"last_processed": index}, f)


def load_status():
    if not os.path.exists("pipeline_status.json"):
        return -1
    with open("pipeline_status.json", "r") as f:
        return json.load(f)["last_processed"]


def process_video(url, index):
    print(f"\nProcessing Video {index}")
    folder = create_video_folder(index)

    video_path = download_video(url, folder)
    audio_path = extract_audio(video_path, folder)
    transcript = transcribe(audio_path)

    transcript_path = os.path.join(folder, "transcript.json")
    save_transcript(transcript, transcript_path)

    chunks = chunk_transcript(transcript)
    chunks_path = os.path.join(folder, "chunks.json")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)

    qa_dataset_path = os.path.join(folder, "qa_dataset.json")
    failed_log_path = os.path.join(folder, "failed_chunks.json")
    generate_dataset(
        input_file=chunks_path,
        output_file=qa_dataset_path,
        failed_log=failed_log_path,
    )

    if os.path.exists(video_path):
        os.remove(video_path)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"Video {index} completed")


def main():
    urls = load_video_list()
    last = load_status()

    print(f"Total Videos: {len(urls)}")
    print(f"Resuming from: {last + 1}")

    for i, url in enumerate(urls):
        if i <= last:
            continue

        start = time.time()

        try:
            process_video(url, i)
            save_status(i)
        except Exception as e:
            print("Error:", str(e))
            print("Stopping pipeline")
            break

        end = time.time()
        print("Time:", round(end - start, 2), "sec")


if __name__ == "__main__":
    main()
