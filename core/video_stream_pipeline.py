import os
import time
import json
import queue
import threading
import subprocess

import yt_dlp
from faster_whisper import WhisperModel

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ================= CONFIG =================

llm_queue = queue.Queue()
MAX_RETRIES = 3
SEGMENT_TIME = 120
WHISPER_WORKERS = 4
OUTPUT_ROOT = "work"

WHISPER_MODEL = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

# ================= GPU MONITOR =================

def get_gpu_stats():
    """Returns (gpu_util_percent, vram_used_mb, vram_total_mb) or None if unavailable."""
    if not NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (
            util.gpu,
            mem.used // (1024 ** 2),
            mem.total // (1024 ** 2)
        )
    except Exception:
        return None


def gpu_monitor_worker(stop_event, interval=5):
    """Background thread that prints GPU utilization and VRAM every `interval` seconds."""
    print("🖥  GPU monitor started")
    while not stop_event.is_set():
        stats = get_gpu_stats()
        if stats:
            gpu_util, vram_used, vram_total = stats
            print(
                f"[GPU] Utilization: {gpu_util:3d}% | "
                f"VRAM: {vram_used} MB / {vram_total} MB "
                f"({vram_used * 100 // vram_total}% used)"
            )
        else:
            print("[GPU] Stats unavailable (pynvml not installed or no NVIDIA GPU)")
        time.sleep(interval)

# ================= DOWNLOAD =================

def download_video(url, work_dir):
    os.makedirs(work_dir, exist_ok=True)

    ydl_opts = {
        "outtmpl": f"{work_dir}/video.%(ext)s",
        "format": "bestvideo[height<=480]+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for f in os.listdir(work_dir):
        if f.startswith("video.") and not f.endswith(".part"):
            return os.path.join(work_dir, f)

    raise RuntimeError("Video download failed")

# ================= SEGMENTER =================

def start_segmentation(video_path, seg_dir):
    os.makedirs(seg_dir, exist_ok=True)

    pattern = os.path.join(seg_dir, "audio_%03d.wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-f", "segment",
        "-segment_time", str(SEGMENT_TIME),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        pattern,
        "-y"
    ]

    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ================= FILE READY CHECK =================

def wait_file_complete(path):
    last_size = -1

    while True:
        size = os.path.getsize(path)
        if size == last_size:
            return
        last_size = size
        time.sleep(1)

# ================= WATCHER =================

def segment_watcher(seg_dir, task_queue, stop_event):
    seen = set()

    while not stop_event.is_set():
        files = sorted([
            f for f in os.listdir(seg_dir)
            if f.endswith(".wav")
        ])

        for f in files:
            full = os.path.join(seg_dir, f)

            if full not in seen:
                wait_file_complete(full)
                print(f"📦 New segment ready: {f}")
                task_queue.put(full)
                seen.add(full)

        time.sleep(2)

# ================= TRANSCRIBER =================

def whisper_worker(task_queue, output_file, stop_event):
    while not stop_event.is_set() or not task_queue.empty():
        try:
            segment_path = task_queue.get(timeout=3)
        except:
            continue

        name = os.path.basename(segment_path)
        idx = int(name.split("_")[1].split(".")[0])
        offset = idx * SEGMENT_TIME

        print(f"🎤 GPU Transcribing segment {idx}")

        segments, _ = WHISPER_MODEL.transcribe(
            segment_path,
            beam_size=1
        )

        with open(output_file, "a", encoding="utf-8") as writer:
            for s in segments:
                record = {
                    "start": s.start + offset,
                    "end": s.end + offset,
                    "text": s.text.strip()
                }
                llm_queue.put(record)

        task_queue.task_done()
        print(f"✅ Segment {idx} transcription complete")

# ================= LLM WORKER =================

def llm_worker(stop_event):
    from ai.qa_generator_parallel import process_chunk

    while not stop_event.is_set() or not llm_queue.empty():
        try:
            record = llm_queue.get(timeout=3)
        except:
            continue

        text = record["text"]

        result = process_chunk(0, {"text": text})

        with open("dataset_stream.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        llm_queue.task_done()

# ================= MAIN =================

def process_video_stream(url, vid):
    work_dir = os.path.join(OUTPUT_ROOT, f"video_{vid}")
    seg_dir = os.path.join(work_dir, "segments")

    os.makedirs(work_dir, exist_ok=True)

    print("⬇ Downloading video...")
    video_path = download_video(url, work_dir)

    print("🎬 Starting segmentation...")
    ffmpeg_proc = start_segmentation(video_path, seg_dir)

    q = queue.Queue()
    stop_event = threading.Event()

    # Start GPU monitor
    gpu_monitor = threading.Thread(
        target=gpu_monitor_worker,
        args=(stop_event,),
        daemon=True
    )
    gpu_monitor.start()

    watcher = threading.Thread(target=segment_watcher, args=(seg_dir, q, stop_event))
    watcher.start()

    workers = []
    for _ in range(WHISPER_WORKERS):
        t = threading.Thread(
            target=whisper_worker,
            args=(q, os.path.join(work_dir, "transcript.jsonl"), stop_event)
        )
        t.start()
        workers.append(t)

    llm_workers = []
    for _ in range(2):
        t = threading.Thread(target=llm_worker, args=(stop_event,))
        t.start()
        llm_workers.append(t)

    ffmpeg_proc.wait()

    print("⏳ Waiting remaining segments transcription...")

    q.join()
    stop_event.set()

    watcher.join()
    for w in workers:
        w.join()

    llm_queue.join()
    for w in llm_workers:
        w.join()

    # Final GPU snapshot
    stats = get_gpu_stats()
    if stats:
        gpu_util, vram_used, vram_total = stats
        print(
            f"\n[GPU] Final stats — Utilization: {gpu_util}% | "
            f"VRAM: {vram_used} MB / {vram_total} MB"
        )

    print("✅ Video fully processed")


if __name__ == "__main__":
    url = input("Enter YouTube URL: ")

    start = time.time()

    process_video_stream(url, 0)

    print("⏱ Total time:", round(time.time() - start, 2), "sec")