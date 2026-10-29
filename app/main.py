"""Course API with embeddings, CIFAR-10 classification, and MNIST generation."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field, StringConstraints

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier, MAX_UPLOAD_BYTES
from app.digit_generator import DigitGenerator

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bigram_model = BigramModel(corpus)
    app.state.embedding_model = EmbeddingModel()
    app.state.image_classifier = ImageClassifier()
    app.state.digit_generator = DigitGenerator()
    yield


app = FastAPI(title="Assignment 3: MNIST GAN and Course API", version="0.3.0", lifespan=lifespan)


@app.get('/generate-digit', responses={200: {'content': {'image/png': {}}}}, response_class=Response)
def generate_digit(
    request: Request,
    count: Annotated[int, Query(ge=1, le=64, description='Number of random digits in a PNG grid')] = 1,
    seed: Annotated[int | None, Query(ge=0, le=4294967295, description='Optional reproducible noise seed')] = None,
):
    """Generate unconditional MNIST-style digits. A specific digit label cannot be selected."""
    png, used_seed = request.app.state.digit_generator.generate_png(count, seed)
    return Response(content=png, media_type='image/png', headers={
        'X-Seed': str(used_seed), 'X-Image-Count': str(count), 'Cache-Control': 'no-store',
    })


@app.get('/gan-info')
def gan_info(request: Request):
    return request.app.state.digit_generator.metadata

Word = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class TextGenerationRequest(BaseModel):
    start_word: Annotated[Word, Field(pattern=r"^\w+$")]
    length: Annotated[int, Field(strict=True, ge=1, le=200)] = 20


class TextGenerationResponse(BaseModel):
    generated_text: str


class EmbeddingResponse(BaseModel):
    word: str
    model: str
    dimensions: int
    embedding: list[float]


class ClassificationResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


@app.post("/classify", response_model=ClassificationResponse)
def classify_image(request: Request, file: Annotated[UploadFile, File(description="PNG or JPEG image, at most 5 MiB")]):
    """Return one of the ten CIFAR-10 classes using the trained assignment CNN."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image upload exceeds 5 MiB.")
    try:
        return request.app.state.image_classifier.predict(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate", response_model=TextGenerationResponse)
def generate_text(body: TextGenerationRequest, request: Request):
    return {"generated_text": request.app.state.bigram_model.generate_text(body.start_word, body.length)}


@app.get("/embedding", response_model=EmbeddingResponse)
def get_embedding(request: Request, word: Annotated[Word, Query(description="One English word", examples=["apple"])]):
    """Return the complete spaCy vector. Unknown vocabulary returns HTTP 404."""
    try:
        return request.app.state.embedding_model.calculate_embedding(word)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
