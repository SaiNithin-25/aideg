import json

INPUT_FILE = "data/transcript.json"
OUTPUT_FILE = "data/chunks.json"


def load_transcript():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)


def chunk_transcript(transcript,
                     max_duration=120,
                     max_chars=800):

    chunks = []
    current_chunk = {
        "start": transcript[0]["start"],
        "end": transcript[0]["end"],
        "text": transcript[0]["text"]
    }

    for seg in transcript[1:]:

        duration = seg["end"] - current_chunk["start"]
        length = len(current_chunk["text"])

        if duration < max_duration and length < max_chars:
            current_chunk["text"] += " " + seg["text"]
            current_chunk["end"] = seg["end"]
        else:
            chunks.append(current_chunk)

            current_chunk = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }

    chunks.append(current_chunk)
    return chunks


def save_chunks(chunks):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(chunks, f, indent=2)


if __name__ == "__main__":
    transcript = load_transcript()
    chunks = chunk_transcript(transcript)

    print(f"Generated {len(chunks)} chunks")

    save_chunks(chunks)