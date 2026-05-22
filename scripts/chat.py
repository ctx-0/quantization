"""
REPL chat for hf models with streaming.

Usage:
    python chat.py --model <model-id-or-local-path>
    python chat.py --model mistralai/Mistral-7B-Instruct-v0.3
    python chat.py --model ./models/my-llm
    python chat.py --model meta-llama/Llama-3.2-3B-Instruct --cpu

Commands inside the REPL:
    /exit   quit
    /reset  clear conversation history
    /info   show current model + device
"""

import argparse
import sys
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


# ── ANSI colours ─────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(model_path: str, force_cpu: bool = False):
    path = Path(model_path)
    source = str(path) if path.exists() else model_path  # local vs hub id

    print(f"{DIM}Loading tokenizer from {source!r}...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(source)

    device_map = "cpu" if force_cpu else "auto"
    dtype = torch.float32 if force_cpu else torch.bfloat16

    print(f"{DIM}Loading model (device_map={device_map!r}, dtype={dtype})...{RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        device_map=device_map,
        torch_dtype=dtype,
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"{GREEN}Model ready on {device}{RESET}\n")
    return model, tokenizer, device


# ── Streaming generation ──────────────────────────────────────────────────────


def stream_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Yield tokens one by one using TextIteratorStreamer."""

    # apply_chat_template works for instruct models that ship a chat template;
    # falls back to a plain concatenation otherwise.
    try:
        result = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Some transformers versions return a BatchEncoding dict, others a raw tensor
        input_ids = (result["input_ids"] if hasattr(result, "keys") else result).to(
            model.device
        )
    except Exception:
        # Fallback: naive user/assistant turns
        prompt = (
            "".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
                for m in messages
            )
            + "Assistant:"
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Run generation in a background thread so we can stream here
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token in streamer:
        yield token

    thread.join()


# ── REPL ──────────────────────────────────────────────────────────────────────

COMMANDS = {"/exit", "/quit", "/reset", "/info"}


def repl(model, tokenizer, device, args):
    model_label = args.model
    history: list[dict] = []

    if args.system:
        history.append({"role": "system", "content": args.system})

    print(
        f"{BOLD}{CYAN}Local LLM Chat{RESET}  {DIM}(type /exit to quit, /reset to clear history){RESET}\n"
    )

    while True:
        # Prompt
        try:
            user_input = input(f"{BOLD}{GREEN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye.{RESET}")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() in ("/exit", "/quit"):
            print(f"{DIM}Bye.{RESET}")
            break

        if user_input.lower() == "/reset":
            history = [
                h for h in history if h["role"] == "system"
            ]  # keep system prompt
            print(f"{YELLOW}History cleared.{RESET}")
            continue

        if user_input.lower() == "/info":
            print(f"  model  : {model_label}")
            print(f"  device : {device}")
            print(f"  turns  : {sum(1 for h in history if h['role'] == 'user')}")
            continue

        history.append({"role": "user", "content": user_input})

        # Stream the response
        print(f"{BOLD}{CYAN}llm>{RESET} ", end="", flush=True)
        reply_parts: list[str] = []

        try:
            for token in stream_response(
                model,
                tokenizer,
                history,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ):
                print(token, end="", flush=True)
                reply_parts.append(token)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[interrupted]{RESET}", flush=True)

        print()  # newline after response
        reply = "".join(reply_parts).strip()

        if reply:
            history.append({"role": "assistant", "content": reply})


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="REPL chat with a local / HuggingFace LLM")
    p.add_argument("--model", required=True, help="HF model id or local folder path")
    p.add_argument("--system", default=None, help="Optional system prompt")
    p.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    p.add_argument(
        "--cpu",
        action="store_true",
        dest="cpu",
        help="Force CPU (default: auto-detect GPU/MPS)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        model, tokenizer, device = load_model(args.model, force_cpu=args.cpu)
    except Exception as e:
        print(f"{RED}Failed to load model: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    repl(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
