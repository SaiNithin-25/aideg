import requests

res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:7b",
        "prompt": "Explain AI simply",
        "stream": False
    }
)

print(res.json()["response"])