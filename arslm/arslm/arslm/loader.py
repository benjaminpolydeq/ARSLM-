from peft import PeftModel
from ARSLM import ARSLM

_cache = {}

def load_model(adapter):
    if adapter in _cache:
        return _cache[adapter]

    base = ARSLM(vocab_size=32000, hidden_size=512)
    model = PeftModel.from_pretrained(base, f"lora/{adapter}")
    model.eval()

    _cache[adapter] = model
    return model