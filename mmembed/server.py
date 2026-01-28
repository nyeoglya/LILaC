import typing as tp
import traceback

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import GenerationInput


model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        from model import MMEmbed
        model = MMEmbed(device="cuda:2")
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")

    print("[Startup] Model ready.")
    yield
    print("[Shutdown] Cleaning up...")
    model = None

app = FastAPI(title="MM-Embed Inference Server", lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    instruction: str
    text: str = ""
    image_path: str = ""

class EmbeddingResponse(BaseModel):
    embedding: tp.List[float]
    dim: int

class BatchEmbeddingRequest(BaseModel):
    items: tp.List[EmbeddingRequest]

class BatchEmbeddingResponse(BaseModel):
    embeddings: tp.List[tp.List[float]]
    dim: int

@app.post("/embed", response_model=EmbeddingResponse)
def embed(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    gen_input = GenerationInput(
        instruction=request.instruction,
        text=request.text,
        image_path=request.image_path,
    )

    try:
        emb = model.get_embeddings([gen_input])
        vec = emb.squeeze(0).tolist()
        return EmbeddingResponse(embedding=vec, dim=len(vec))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
def embed_batch(request: BatchEmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    gen_inputs = [
        GenerationInput(
            instruction=i.instruction,
            text=i.text,
            image_path=i.image_path,
        ) 
        for i in request.items
    ]

    try:
        embeddings = model.get_embeddings(gen_inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    vecs = embeddings.tolist()
    return BatchEmbeddingResponse(
        embeddings=vecs,
        dim=len(vecs[0]) if vecs else 0,
    )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
    )
