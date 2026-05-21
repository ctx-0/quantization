# quants


## models

### qwen2-0.5b

*repo*: https://huggingface.co/Qwen/Qwen2-0.5B
```sh
hf download Qwen/Qwen2-0.5B --local-dir ./qwen2-0.5b
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
python llama.cpp/convert_hf_to_gguf.py ./qwen2-0.5b --outtype f16 --outfile ./qwen2-0.5b-f16.gguf

# quantize F16 → Q4_K_M
./llama.cpp/llama-quantize ./qwen2-0.5b-f16.gguf ./qwen2-0.5b-Q4_K_M.gguf Q4_K_M
```

quick test
```sh
./llama.cpp/llama-cli \
    -m ./qwen2-0.5b-q4km.gguf \
    -p "The capital of France is" \
    -n 20
```

*official hf space*: https://huggingface.co/spaces/ggml-org/gguf-my-repo

#### perplexity

*TODO*

```sh
./llama.cpp/llama-perplexity -m ./qwen2-0.5b-Q4_K_M.gguf wikitext-2-raw/wiki.test.raw
```


## evaluating

*TODO*

`lm-eval-harness`
