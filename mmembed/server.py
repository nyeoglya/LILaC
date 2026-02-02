import typing as tp
import traceback

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import GenerationInput, QueryGenerationInput

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
    text: str = ""
    img_path: str = ""

class QueryEmbeddingRequest(BaseModel):
    instruction: str = ""
    text: str = ""
    img_path: str = ""

class EmbeddingResponse(BaseModel):
    embedding: tp.List[float]

class BatchEmbeddingRequest(BaseModel):
    items: tp.List[EmbeddingRequest]

class BatchEmbeddingResponse(BaseModel):
    embeddings: tp.List[tp.List[float]]

@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    gen_input = GenerationInput(
        text=request.text,
        img_path=request.img_path,
    )

    try:
        emb = model.embedding(gen_input)
        vec = emb.squeeze(0).tolist()
        return EmbeddingResponse(embedding=vec)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/query", response_model=EmbeddingResponse)
async def embed(request: QueryEmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    query_gen_input = QueryGenerationInput(
        instruction=request.instruction,
        text=request.text,
        img_path=request.img_path,
    )

    try:
        emb = model.query_embedding(query_gen_input)
        vec = emb.squeeze(0).tolist()
        return EmbeddingResponse(embedding=vec)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_batch(request: BatchEmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    gen_inputs = [GenerationInput(
        text=i.text,
        img_path=i.img_path,
    ) for i in request.items]

    try:
        embeddings = model.batch_embedding(gen_inputs)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    vecs = [emb.squeeze(0).tolist() for emb in embeddings]
    return BatchEmbeddingResponse(embeddings=vecs)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
    )
