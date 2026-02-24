#!/usr/bin/env python3
"""
Interactive chat with Gemma 3 1B (Q4 / Q8) using llama-cpp-python.

Usage:
    python chat.py                    # uses Q4 model by default
    python chat.py --quant q8         # uses Q8 model
    python chat.py --model path.gguf  # uses a custom GGUF file

The script will automatically download the model if it is not found locally.
"""

import argparse
import glob
import os
import sys
import textwrap

from llama_cpp import Llama

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MODELS_DIR = "models"
DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher, freundlicher KI-Assistent. "
    "Antworte präzise und auf Deutsch, sofern der Nutzer nicht anders fragt."
)

# Generation parameters
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.1

# Context window (Gemma 3 1B supports up to 32 k)
DEFAULT_CTX_SIZE = 8192


# ── Helper ──────────────────────────────────────────────────────────────────

def resolve_model_path(quant: str, models_dir: str) -> str:
    """Return the path to a GGUF file, downloading it if necessary."""
    pattern_map = {
        "q4": "*Q4*.gguf",
        "q8": "*Q8*.gguf",
    }
    pattern = os.path.join(models_dir, pattern_map.get(quant, "*.gguf"))
    matches = sorted(glob.glob(pattern))

    if matches:
        return matches[0]

    # Model not found → try downloading
    print(f"[INFO] No {quant.upper()} model found in '{models_dir}'. Downloading …")
    try:
        from download_model import download
        return download(quant, models_dir)
    except Exception as exc:
        print(f"[ERROR] Could not download model: {exc}")
        print("Please run:  python download_model.py --quant", quant)
        sys.exit(1)


def build_prompt_messages(history: list[dict], system_prompt: str) -> list[dict]:
    """Build the messages list in chat-ML style for create_chat_completion."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    return messages


def format_assistant_text(text: str, width: int = 88) -> str:
    """Wrap long assistant replies for nicer terminal output."""
    paragraphs = text.split("\n")
    wrapped = []
    for para in paragraphs:
        if para.strip():
            wrapped.append(textwrap.fill(para, width=width))
        else:
            wrapped.append("")
    return "\n".join(wrapped)


# ── Main Chat Loop ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chat with Gemma 3 1B (GGUF, quantized)"
    )
    parser.add_argument(
        "--quant",
        choices=["q4", "q8"],
        default="q4",
        help="Quantization to use (default: q4)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a specific GGUF model file (overrides --quant)",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="Directory where models are stored (default: models/)",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the assistant",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=DEFAULT_CTX_SIZE,
        help=f"Context window size (default: {DEFAULT_CTX_SIZE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate per reply (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all, 0 = CPU only)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (wait for full response)",
    )
    args = parser.parse_args()

    # ── Resolve model path ──────────────────────────────────────────────────
    if args.model:
        model_path = args.model
    else:
        model_path = resolve_model_path(args.quant, args.models_dir)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    print(f"[INFO] Loading model: {model_path}")
    print(f"[INFO] Context size : {args.ctx_size}")
    print(f"[INFO] GPU layers   : {args.gpu_layers}")
    print()

    # ── Load model ──────────────────────────────────────────────────────────
    llm = Llama(
        model_path=model_path,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        verbose=False,
    )

    # ── Chat loop ───────────────────────────────────────────────────────────
    history: list[dict] = []
    stream = not args.no_stream

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         Gemma 3 1B Chat  –  Lokaler KI-Assistent           ║")
    print("║                                                            ║")
    print("║  Befehle:                                                  ║")
    print("║    /clear   – Chatverlauf zurücksetzen                     ║")
    print("║    /system  – System-Prompt ändern                         ║")
    print("║    /quit    – Beenden  (oder Ctrl+C / Ctrl+D)              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    system_prompt = args.system_prompt

    while True:
        # Read user input
        try:
            user_input = input("\033[1;34m Du:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Bye!]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.lower() == "/quit":
            print("[Bye!]")
            break
        if user_input.lower() == "/clear":
            history.clear()
            print("[Chat-Verlauf gelöscht.]\n")
            continue
        if user_input.lower().startswith("/system"):
            new_prompt = user_input[len("/system"):].strip()
            if new_prompt:
                system_prompt = new_prompt
                print(f"[System-Prompt gesetzt: {system_prompt}]\n")
            else:
                print(f"[Aktueller System-Prompt: {system_prompt}]\n")
            continue

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Build messages
        messages = build_prompt_messages(history, system_prompt)

        # Generate response
        print("\033[1;32m KI:\033[0m ", end="", flush=True)

        if stream:
            # Streaming mode – tokens appear as they are generated
            response_text = ""
            for chunk in llm.create_chat_completion(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=DEFAULT_TOP_P,
                repeat_penalty=DEFAULT_REPEAT_PENALTY,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    response_text += token
            print()  # newline after streaming
        else:
            # Non-streaming mode
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=DEFAULT_TOP_P,
                repeat_penalty=DEFAULT_REPEAT_PENALTY,
            )
            response_text = result["choices"][0]["message"]["content"]
            print(format_assistant_text(response_text))

        # Add assistant reply to history
        history.append({"role": "assistant", "content": response_text})
        print()


if __name__ == "__main__":
    main()
