import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import typing as tp

from graph.graph import LILaCGraph

app = FastAPI(title="LILaC Graph Server")

GRAPH_PATH = ""
lilac_graph = None

class SingleRequest(BaseModel):
    prompt: str
    image_path: tp.Optional[str] = None
    max_tokens: tp.Optional[int] = 512
    system_prompt: tp.Optional[str] = None

@app.on_event("startup")
def load_model():
    global lilac_graph
    print(f"Loading predefined graph from {GRAPH_PATH}...")
    try:
        lilac_graph = LILaCGraph(graph_path=GRAPH_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
def health_check():
    if lilac_graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")
    return {"status": "ok", "model": GRAPH_PATH}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
