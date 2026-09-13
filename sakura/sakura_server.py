import os
import time
from pathlib import Path

_SAKURA_DIR = Path(__file__).resolve().parent

os.environ["PATH"] = str(_SAKURA_DIR / "llama") + os.pathsep + os.environ.get("PATH", "")

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import asyncio

from llama_cpp import Llama

MODEL_PATH = str(_SAKURA_DIR / "models" / "sakura-7b-qwen2.5-v1.0-q6k.gguf")
N_CTX = 2048
# Partial GPU offload: the 7B Q6K model plus a 2048-token KV cache exceed
# the 8GB laptop GPU at n_gpu_layers=99 (llama_decode -1 / CUDA pool
# asserts under concurrent translation load).  Half offload keeps ~3.5GB
# on the GPU with headroom; the rest runs on the CPU.
N_GPU_LAYERS = int(os.environ.get("SAKURA_GPU_LAYERS", "40"))

print("loading sakura model...", flush=True)
t0 = time.time()
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

app = FastAPI(title="sakura-local")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "sukinishiro"
    messages: List[Message]
    temperature: float = 0.3
    top_p: float = 0.3
    max_tokens: int = 512
    frequency_penalty: float = 0.1
    seed: Optional[int] = -1
    extra_query: Optional[Any] = None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    def _run():
        return llm.create_chat_completion(
            messages=[m.model_dump() for m in req.messages],
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            frequency_penalty=req.frequency_penalty,
        )

    out = await asyncio.get_event_loop().run_in_executor(None, _run)
    content = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    return {
        "id": "chatcmpl-sakura-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)},
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "sukinishiro", "object": "model", "owned_by": "local"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
