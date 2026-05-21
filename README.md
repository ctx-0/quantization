# quantization


## models

### qwen3.5-9b

*repo*: https://huggingface.co/Qwen/Qwen3.5-9B

### qwen2.5-3b

*repo*: https://huggingface.co/Qwen/Qwen2.5-3B
```sh
hf download Qwen/Qwen2.5-3B --local-dir ./qwen2.5-3b
```

## quick setup

```sh
uv venv --python 3.13
uv sync
python -c "import torch; print(torch.cuda.is_available())"
```

## quantizing

### gguf

*docs*: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md

quantize in two steps
```sh
# convert HF → F16 GGUF (lossless, intermediate step)
python llama.cpp/convert_hf_to_gguf.py ./qwen2.5-3b --outtype f16 --outfile ./qwen2.5-3b-f16.gguf

# quantize F16 → Q4_K_M
./llama.cpp/llama-quantize ./qwen2.5-3b-f16.gguf ./qwen2.5-3b-Q4_K_M.gguf Q4_K_M
```

quick test
```sh
./llama.cpp/llama-cli -m ./qwen2.5-3b-Q4_K_M.gguf -p "The capital of France is" -n 20
```

*official hf space*: https://huggingface.co/spaces/ggml-org/gguf-my-repo

### NF4

quantize and save
```sh
# from local path
python ./bnb.py ./qwen2.5-3b ./qwen2.5-3b-nf4

# from HF
python ./bnb.py Qwen/Qwen2.5-3B ./Qwen2.5-3B-bnb-4bit
```

loads the model, applies 4-bit NF4 with double quantization (bf16 compute), and saves to the output dir.

### AWQ

*TODO*

## evaluating


`lm-eval-harness`

### wikitext (perplexity)

measures quantization degradation — compare perplexity across base, NF4, and Q4_K_M; the delta matters, not the absolute value.

> [!NOTE]
> **OOM**: wikitext uses a rolling window over the full test corpus. Qwen's default context is 32k tokens which blows up VRAM even on small models. cap with `max_length=2048` — results stay comparable as long as it's consistent across runs.

```sh
# base model (baseline)
lm_eval --model hf --model_args pretrained=./qwen2.5-3b,max_length=2048 --tasks wikitext --device cuda

# BnB NF4
lm_eval --model hf --model_args pretrained=./qwen2.5-3b-nf4,max_length=2048 --tasks wikitext --device cuda

# GGUF via HF backend (no server needed)
lm_eval --model hf --model_args pretrained=./qwen2.5-3b,gguf_file=qwen2.5-3b-Q4_K_M.gguf,max_length=2048 --tasks wikitext --device cuda

# GGUF via llama.cpp server (start server first)
lm_eval --model gguf --model_args base_url=http://localhost:8080,tokenizer=./qwen2.5-3b --tasks wikitext
```

## references

1. QLoRA (NF4, double quantization) — Dettmers et al., 2023: https://arxiv.org/abs/2305.14314
2. LLM.int8() — Dettmers et al., 2022: https://arxiv.org/abs/2208.07339
3. AWQ — Lin et al., 2023: https://arxiv.org/abs/2306.00978
4. Qwen2.5-3B: https://huggingface.co/Qwen/Qwen2.5-3B
5. Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B
