import asyncio
import typing as tp
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils import GenerationInput

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from model import Qwen3_VL
        app.state.model = Qwen3_VL(device="cuda:0")
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")

    print("[Startup] Model ready.")
    yield
    print("[Shutdown] Cleaning up...")
    app.state.model = None


app = FastAPI(
    title="Qwen3-VL Inference Server",
    lifespan=lifespan,
)
app.state.sem = asyncio.Semaphore(1)

class SinglePromptRequest(BaseModel):
    text: str
    img_paths: tp.List[str] = Field(default_factory=list)
    max_tokens: int = Field(default=256, ge=1, le=4096)

class SinglePromptResponse(BaseModel):
    response: str


@app.post("/generate", response_model=SinglePromptResponse)
async def generate_text(request: SinglePromptRequest):
    try:
        gen_input = GenerationInput(text=request.text, img_paths=request.img_paths)

        async with app.state.sem:
            output = await asyncio.to_thread(
                app.state.model.inference,
                gen_input,
                request.max_tokens,
            )

        return SinglePromptResponse(response=output.text)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
