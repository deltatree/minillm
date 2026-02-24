#!/usr/bin/env python3
"""
OpenAI-kompatibler HTTP-Server für das lokale Gemma 3 1B Modell.

Startet einen FastAPI-Server, der die OpenAI Chat Completions API nachbildet.
Dadurch kann jedes Tool, das die OpenAI-API unterstützt, gegen dieses lokale
Modell arbeiten (z.B. curl, Python openai-Library, LangChain, etc.).

Usage:
    python server.py                        # Standard (Q4, Port 8000)
    python server.py --quant q8             # Q8-Modell
    python server.py --port 9000            # anderer Port
    python server.py --model path.gguf      # eigenes Modell
"""

import argparse
import os
import sys
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from llama_cpp import Llama

# ── Import helpers from chat.py ─────────────────────────────────────────────
from chat import (
    resolve_model_path,
    DEFAULT_MODELS_DIR,
    DEFAULT_CTX_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SYSTEM_PROMPT,
)

# ── Pydantic Models (OpenAI-kompatibel) ────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3-1b-it"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    repeat_penalty: Optional[float] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── Globals ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MiniLLM – OpenAI-kompatible API",
    description="Lokaler LLM-Server mit OpenAI-kompatibler REST-Schnittstelle",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm: Optional[Llama] = None
model_name: str = "gemma-3-1b-it"


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/v1/models")
def list_models() -> ModelListResponse:
    """Liste der verfügbaren Modelle (OpenAI-kompatibel)."""
    return ModelListResponse(
        data=[ModelInfo(id=model_name)]
    )


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """OpenAI-kompatibler Chat-Completions-Endpoint."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    max_tokens = request.max_tokens or DEFAULT_MAX_TOKENS
    temperature = request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
    top_p = request.top_p if request.top_p is not None else DEFAULT_TOP_P
    repeat_penalty = request.repeat_penalty if request.repeat_penalty is not None else DEFAULT_REPEAT_PENALTY

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming ───────────────────────────────────────────────────────
    if request.stream:
        def generate():
            import json

            for chunk in llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                finish_reason = chunk["choices"][0].get("finish_reason")

                sse_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token} if token else {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(sse_data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── Non-Streaming ───────────────────────────────────────────────────
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )

    choice = result["choices"][0]
    usage = result.get("usage", {})

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=choice["message"]["content"],
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


@app.get("/health")
def health():
    """Health-Check-Endpoint."""
    return {"status": "ok", "model": model_name, "loaded": llm is not None}


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    global llm, model_name

    parser = argparse.ArgumentParser(
        description="OpenAI-kompatibler API-Server für lokales LLM"
    )
    parser.add_argument(
        "--quant", choices=["q4", "q8"], default="q4",
        help="Quantisierung (default: q4)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Pfad zu einer GGUF-Datei (überschreibt --quant)",
    )
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR,
        help="Verzeichnis für Modelle (default: models/)",
    )
    parser.add_argument(
        "--ctx-size", type=int, default=DEFAULT_CTX_SIZE,
        help=f"Kontextfenstergröße (default: {DEFAULT_CTX_SIZE})",
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=-1,
        help="GPU-Layer (-1 = alle, 0 = nur CPU)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host-Adresse (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port (default: 8000)",
    )
    args = parser.parse_args()

    # ── Modell laden ────────────────────────────────────────────────────
    if args.model:
        model_path = args.model
    else:
        model_path = resolve_model_path(args.quant, args.models_dir)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Modelldatei nicht gefunden: {model_path}")
        sys.exit(1)

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print(f"[INFO] Lade Modell: {model_path}")
    print(f"[INFO] Kontextgröße: {args.ctx_size}")
    print(f"[INFO] GPU-Layer: {args.gpu_layers}")

    llm = Llama(
        model_path=model_path,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        verbose=False,
    )

    print(f"\n[INFO] Server startet auf http://{args.host}:{args.port}")
    print(f"[INFO] OpenAI-Base-URL: http://localhost:{args.port}/v1")
    print(f"[INFO] Docs: http://localhost:{args.port}/docs\n")

    # ── Server starten ─────────────────────────────────────────────────
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
