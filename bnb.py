"""quantize to NF4"""

import torch
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = ArgumentParser()
parser.add_argument("model_src", help="HF repo id or local path")
parser.add_argument("save_dir", help="output directory")
args = parser.parse_args()

MODEL_SRC = args.model_src
SAVE_DIR = args.save_dir


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
