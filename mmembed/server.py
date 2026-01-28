import typing as tp
import uvicorn
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import MMEmbed
from utils import GenerationInput


# ======================
# Config
# ======================
MODEL_PATH = "nvidia/MM-Embed"
BATCH_SIZE = 1  # mm-embed는 batch 키워도 VRAM 급증


# ======================
# App
# ======================
app = FastAPI(title="NVIDIA MM-Embed Inference Server")

model: tp.Optional[MMEmbed] = None


# ======================
# Schemas
# ======================
class EmbeddingRequest(BaseModel):
    prompt: str


class EmbeddingResponse(BaseModel):
    embedding: tp.List[float]
    dim: int


class BatchItem(BaseModel):
    prompt: str


class BatchEmbeddingRequest(BaseModel):
    items: tp.List[BatchItem]


class BatchEmbeddingResponse(BaseModel):
    embeddings: tp.List[tp.List[float]]
    dim: int


# ======================
# Lifecycle
# ======================
@app.on_event("startup")
def load_model():
    global model
    print(f"[Startup] Loading model: {MODEL_PATH}")

    try:
        model = MMEmbed(
            model_path=MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float32,
        )
    except Exception as e:
        # ❌ 서버만 살아있는 상태 방지
        raise RuntimeError(f"Model load failed: {e}")

    print("[Startup] Model ready.")


# ======================
# Endpoints
# ======================
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "ok",
        "model": MODEL_PATH,
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }


@app.post("/embed", response_model=EmbeddingResponse)
def embed(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    gen_input = GenerationInput(text_prompt=request.prompt)

    try:
        embeddings = model.get_embeddings([gen_input], batch_size=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    vec = embeddings[0].tolist()
    return EmbeddingResponse(embedding=vec, dim=len(vec))


@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
def embed_batch(request: BatchEmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    gen_inputs = [
        GenerationInput(text_prompt=item.prompt)
        for item in request.items
    ]

    try:
        embeddings = model.get_embeddings(gen_inputs, batch_size=BATCH_SIZE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    vectors = embeddings.tolist()
    return BatchEmbeddingResponse(
        embeddings=vectors,
        dim=len(vectors[0]) if vectors else 0,
    )


# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    # ⚠️ 반드시 workers=1
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
    )
