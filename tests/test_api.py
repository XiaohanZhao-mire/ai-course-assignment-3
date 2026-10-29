import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.bigram_model import BigramModel


@pytest.fixture(scope="module")
def client():
    # This loads the actual en_core_web_lg model, without mock embeddings.
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("word", ["apple", "orange", "computer"])
def test_embedding_matches_course_calculation(client, word):
    response = client.get("/embedding", params={"word": word})
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == word
    assert data["model"] == "en_core_web_lg"
    assert data["dimensions"] == len(data["embedding"]) == 300
    assert all(math.isfinite(x) for x in data["embedding"])
    assert any(x != 0 for x in data["embedding"])
    np.testing.assert_allclose(data["embedding"], app.state.embedding_model.nlp(word).vector)


def test_words_have_different_vectors(client):
    apple = client.get("/embedding", params={"word": "apple"}).json()["embedding"]
    car = client.get("/embedding", params={"word": "car"}).json()["embedding"]
    assert apple != car


@pytest.mark.parametrize("word", ["", "   ", "two words", "123", "!!!", "a" * 101])
def test_invalid_word(client, word):
    assert client.get("/embedding", params={"word": word}).status_code == 422


def test_missing_word(client):
    assert client.get("/embedding").status_code == 422


def test_unknown_word(client):
    assert client.get("/embedding", params={"word": "zzzxqvzzzxqvzzzxqv"}).status_code == 404


def test_whitespace_is_trimmed(client):
    assert client.get("/embedding", params={"word": " apple "}).json()["word"] == "apple"


def test_original_endpoints(client):
    assert client.get("/").json() == {"Hello": "World"}
    response = client.post("/generate", json={"start_word": "generating", "length": 4})
    assert response.status_code == 200
    assert response.json() == {"generated_text": "generating text based on"}


@pytest.mark.parametrize("length", [0, -1, 201, 1.5, True])
def test_invalid_generation_length(client, length):
    assert client.post("/generate", json={"start_word": "bigram", "length": length}).status_code == 422


def test_bigram_stops_at_dead_end_and_respects_length():
    model = BigramModel(["a b a", "c d"])
    assert model.generate_text("a", 1) == "a"
    assert model.generate_text("missing", 20) == "missing"
    assert model.generate_text("c", 20) == "c d"
    assert model.probabilities["a"] == {"b": 1.0}
