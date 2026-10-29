"""Exercise an actual running API, including the Docker container in CI."""
import json
import math
import sys
import struct
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def get(path):
    with urlopen(base + path, timeout=30) as response:
        return json.load(response)


assert get("/") == {"Hello": "World"}
apple = get("/embedding?word=apple")
orange = get("/embedding?word=orange")
for data in [apple, orange]:
    assert data["model"] == "en_core_web_lg"
    assert data["dimensions"] == len(data["embedding"]) == 300
    assert all(math.isfinite(x) for x in data["embedding"])
    assert any(x != 0 for x in data["embedding"])
assert apple["embedding"] != orange["embedding"]
for path, code in [("/embedding", 422), ("/embedding?word=two%20words", 422),
                   ("/embedding?word=zzzxqvzzzxqvzzzxqv", 404)]:
    try:
        get(path)
        raise AssertionError(f"Expected HTTP {code}: {path}")
    except HTTPError as exc:
        assert exc.code == code
body = json.dumps({"start_word": "generating", "length": 4}).encode()
with urlopen(Request(base + "/generate", data=body, headers={"Content-Type": "application/json"}), timeout=30) as response:
    assert json.load(response) == {"generated_text": "generating text based on"}
print(json.dumps({"status": "PASS", "dimensions": apple["dimensions"],
                  "apple_first_5": apple["embedding"][:5],
                  "checks": "real embeddings, distinct words, input errors, original generation"}, indent=2))

# Multipart upload using the standard library, so this also runs in CI.
root = Path(__file__).resolve().parent.parent
boundary = "Assignment2ImageBoundary"
examples = json.loads((root / "examples/labels.json").read_text())
results = []
for example in examples:
    image = (root / "examples" / example["file"]).read_bytes()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"example.png\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + image + f"\r\n--{boundary}--\r\n".encode()
    request = Request(base + "/classify", data=body,
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert 0 <= result["class_id"] < 10
    assert result["class_name"] in result["probabilities"]
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 10
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    results.append({"file": example["file"], "true_class": example["true_class_name"],
                    "predicted_class": result["class_name"], "confidence": result["confidence"]})
assert len({r["predicted_class"] for r in results}) > 1
print(json.dumps({"status": "PASS", "image_classification_examples": results}, indent=2))

# Real GAN inference over HTTP. Decode PNG dimensions without extra dependencies.
def get_png(query):
    with urlopen(base + '/generate-digit?' + query, timeout=30) as response:
        assert response.headers['Content-Type'] == 'image/png'
        content = response.read()
        assert content[:8] == b'\x89PNG\r\n\x1a\n'
        return content, response.headers['X-Seed']

single, seed = get_png('seed=42')
assert seed == '42' and struct.unpack('>II', single[16:24]) == (28,28)
assert single == get_png('seed=42')[0]
assert single != get_png('seed=43')[0]
grid, _ = get_png('count=64&seed=2026')
assert struct.unpack('>II', grid[16:24]) == (224,224)
for query in ['count=0', 'count=65', 'seed=-1', 'seed=abc']:
    try:
        get_png(query)
        raise AssertionError('Expected invalid GAN input to return HTTP 422')
    except HTTPError as exc:
        assert exc.code == 422
metadata = get('/gan-info')
assert metadata['training_images'] == 60000 and metadata['epochs'] > 0
print(json.dumps({'status': 'PASS', 'gan': 'trained MNIST checkpoint',
                  'epochs': metadata['epochs'], 'checks': 'PNG, grid, seeds, invalid inputs'}, indent=2))
