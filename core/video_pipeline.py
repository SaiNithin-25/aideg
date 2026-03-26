import os
from runtime_bootstrap import ensure_local_site_packages

ensure_local_site_packages()

import yt_dlp
import subprocess
from faster_whisper import WhisperModel

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_video(url, work_dir):
    os.makedirs(work_dir, exist_ok=True)

    ydl_opts = {
        'outtmpl': f'{work_dir}/video.%(ext)s',
        'format': 'bestvideo[height<=480]+bestaudio/best',
        'merge_output_format': 'mp4',
        'js_runtimes': {'node': {}}
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for filename in os.listdir(work_dir):
        if filename.startswith("video.") and not filename.endswith(".part"):
            return os.path.join(work_dir, filename)

    raise FileNotFoundError(f"No downloaded video file found in {work_dir}")


def stream_audio_segments(video_path, work_dir, segment_time=300):
    segment_dir = os.path.join(work_dir, "segments")
    os.makedirs(segment_dir, exist_ok=True)

    output_pattern = os.path.join(segment_dir, "audio_%03d.wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        output_pattern,
        "-y"
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    segments = sorted([
        os.path.join(segment_dir, f)
        for f in os.listdir(segment_dir)
        if f.endswith(".wav")
    ])

    return segments


def transcribe_segments(segments):
    model = WhisperModel("medium", device="cuda")
    all_transcripts = []

    for segment in segments:
        print(f"Transcribing {segment}...")
        segs, _ = model.transcribe(segment)

        for s in segs:
            chunk = {
                "start": s.start,
                "end": s.end,
                "text": s.text
            }
            all_transcripts.append(chunk)

            # 🚀 Stream each chunk immediately to your LLM queue
            # send_to_llm_queue(chunk)

    return all_transcripts


def save_transcript(transcript, output_path=None):
    import json

    if output_path is None:
        output_path = f"{OUTPUT_DIR}/transcript.json"

    with open(output_path, "w") as f:
        json.dump(transcript, f, indent=2)


if __name__ == "__main__":
    url = input("Enter YouTube URL: ")

    print("Downloading video...")
    video = download_video(url, OUTPUT_DIR)

    print("Splitting audio into segments...")
    segments = stream_audio_segments(video, OUTPUT_DIR, segment_time=300)

    print("Transcribing segments...")
    transcript = transcribe_segments(segments)

    print("Saving transcript...")
    save_transcript(transcript)

    print("DONE ✅")