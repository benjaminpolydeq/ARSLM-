import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./arslm_final"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

def chat(text):
    prompt = (
        "You are ARSLM, an intelligent and friendly AI assistant.\n"
        f"User: {text}\nARSLM:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).split("ARSLM:")[-1]

while True:
    q = input("\nUser: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("ARSLM:", chat(q))