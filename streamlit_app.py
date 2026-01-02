# ================================
# streamlit_app.py – ARSLM isolé
# ================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 🔹 Chemins vers ton modèle LoRA
MODEL_PATH = "./arslm_lora"
TOKENIZER_PATH = "./arslm_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Charger le tokenizer et le modèle
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# 🔹 Fonction pour générer des réponses
def generate_response(prompt_text, max_length=200, temperature=0.8, top_p=0.9):
    prompt = f"You are ARSLM, an intelligent English assistant.\nUser: {prompt_text}\nARSLM:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("ARSLM:")[-1].strip()
    return response

# ================================
# 🔹 Streamlit UI
# ================================
st.set_page_config(page_title="ARSLM Test Interface", layout="centered")
st.title("🤖 ARSLM – Test English Responses")

user_input = st.text_area("Enter your message:", "")

if st.button("Send"):
    if user_input.strip() != "":
        with st.spinner("Generating response..."):
            reply = generate_response(user_input)
        st.markdown(f"**ARSLM:** {reply}")
    else:
        st.warning("Please enter a message to send.")