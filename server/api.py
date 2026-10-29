
import asyncio
from contextlib import asynccontextmanager
from queue import Full

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from server.runtime import TinyServeRuntime, MODEL_ID

import json
from fastapi.responses import StreamingResponse


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=64)


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    generated_token_ids: list[int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = TinyServeRuntime()
    await asyncio.to_thread(runtime.start)
    app.state.runtime = runtime

    try:
        yield
    finally:
        await asyncio.to_thread(runtime.stop)


app = FastAPI(title="TinyServe", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request):
    runtime = request.app.state.runtime

    try:
        future = runtime.submit(body.prompt, body.max_new_tokens)
    except Full:
        raise HTTPException(503, "Request queue is full")
    except RuntimeError:
        raise HTTPException(503, "Inference worker is unavailable")

    try:
        return await asyncio.shield(asyncio.wrap_future(future))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(500, "Inference failed")


@app.post("/generate/stream")
async def generate_stream(body: GenerateRequest, request: Request):
    runtime = request.app.state.runtime

    try:
        _, events = runtime.submit_stream(
            body.prompt, body.max_new_tokens
        )
    except Full:
        raise HTTPException(503, "Request queue is full")
    except RuntimeError:
        raise HTTPException(503, "Inference worker is unavailable")

    async def event_generator():
        while True:
            item = await asyncio.to_thread(events.get)

            event_type = item["event"]
            payload = json.dumps(item, ensure_ascii=False)

            yield f"event: {event_type}\ndata: {payload}\n\n"

            if event_type in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )