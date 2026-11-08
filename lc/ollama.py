"""Minimal Ollama client (local inference only, no API keys)."""
import json
import urllib.request

URL = "http://localhost:11434/api/generate"


def generate(model, prompt, temperature, seed, num_predict=8, num_ctx=4096, url=URL, raw=False):
    """raw=True sends the prompt as plain text completion (no chat template)."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": raw,
        "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["response"]
