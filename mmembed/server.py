import asyncio
import traceback
import typing as tp

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import GenerationInput, QueryGenerationInput

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from model import MMEmbed
        app.state.model = MMEmbed(device="cuda:1")
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")

    print("[Startup] Model ready.")
    yield
    print("[Shutdown] Cleaning up...")
    app.state.model = None

app = FastAPI(
    title="MM-Embed Inference Server",
    lifespan=lifespan
)
app.state.sem = asyncio.Semaphore(1)

class EmbeddingRequest(BaseModel):
    text: str = ""
    img_path: str = ""

class QueryEmbeddingRequest(BaseModel):
    instruction: str = ""
    text: str = ""
    img_path: str = ""

class EmbeddingResponse(BaseModel):
    embedding: tp.List[float]

@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    gen_input = GenerationInput(
        text=request.text,
        img_path=request.img_path,
    )

    try:
        async with app.state.sem:
            emb = await asyncio.to_thread(
                app.state.model.embedding,
                gen_input,
            )
        vec = emb.squeeze(0).tolist()
        return EmbeddingResponse(embedding=vec)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/query", response_model=EmbeddingResponse)
async def embed(request: QueryEmbeddingRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    query_gen_input = QueryGenerationInput(
        instruction=request.instruction,
        text=request.text,
        img_path=request.img_path,
    )

    try:
        async with app.state.sem:
            emb = await asyncio.to_thread(
                app.state.model.query_embedding,
                query_gen_input,
            )
        vec = emb.squeeze(0).tolist()
        return EmbeddingResponse(embedding=vec)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
