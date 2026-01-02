import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "distilgpt2"
LORA_PATH = "./arslm_lora"
OUTPUT_PATH = "./arslm_final"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD BASE MODEL
# =========================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# =========================
# LOAD LORA
# =========================
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# =========================
# 🔥 MERGE LORA
# =========================
print("🔗 Merging LoRA into base model...")
model = model.merge_and_unload()

# =========================
# SAVE FINAL MODEL
# =========================
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ ARSLM FINAL READY")
print(f"📦 Saved to: {OUTPUT_PATH}")