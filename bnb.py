"""quantize to NF4"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# MODEL_SRC  = "Qwen/Qwen2-0.5B"
MODEL_SRC = "./qwen2-0.5b"
SAVE_DIR = "./qwen2-0.5b-nf4"

# -- Quantize --
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # quantize the quantization constants too
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in nf4
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_SRC,
    quantization_config=bnb_config,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_SRC)

# -- Save --
# saves quantized weights + embeds quantization_config in config.json
model.save_pretrained(SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(SAVE_DIR)
