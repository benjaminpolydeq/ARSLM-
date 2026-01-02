# ================================
# 🔹 streamlit_app.py – ARSLM LoRA
# ================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================================
# ⚙️ CONFIGURATION
# ================================
MODEL_PATH = "./arslm_lora"        # chemin vers ton modèle fine-tuné avec LoRA
TOKENIZER_PATH = "./arslm_lora"    # chemin du tokenizer correspondant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# 🔹 Charger modèle et tokenizer
# ================================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Charger le modèle de base + LoRA
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ================================
# 🔹 Fonction de génération de réponse
# ================================
def generate_response(user_input, max_length=200, temperature=0.8, top_p=0.9):
    prompt = f"You are ARSLM, an intelligent and friendly assistant that speaks English.\nUser: {user_input}\nARSLM:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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
# 🔹 Interface Streamlit
# ================================
st.title("🤖 ARSLM – MicroLLM SaaS (LoRA)")
st.write("ARSLM est un assistant intelligent entraîné en anglais.")

user_input = st.text_input("💬 Pose une question à ARSLM :")

if st.button("Envoyer") and user_input.strip() != "":
    with st.spinner("ARSLM réfléchit..."):
        answer = generate_response(user_input)
        st.markdown(f"**ARSLM:** {answer}")