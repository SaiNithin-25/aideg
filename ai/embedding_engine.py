import json
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/chunks.json"
OUTPUT_FILE = "data/chunks_with_embeddings.json"


def load_chunks():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)


def generate_embeddings(chunks):

    model = SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        device="cuda"
    )

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    return chunks


def save_chunks(chunks):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(chunks, f)


if __name__ == "__main__":
    chunks = load_chunks()

    print("Generating embeddings...")
    chunks = generate_embeddings(chunks)

    save_chunks(chunks)

    print("Done ✅")