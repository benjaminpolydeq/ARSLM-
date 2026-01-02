# ================================
# streamlit_app.py – ARSLM SaaS
# ================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================================
# ⚙️ CONFIGURATION DES CHEMINS
# ================================
BASE_MODEL_PATH = "./distilgpt2"      # modèle de base
LORA_MODEL_PATH = "./arslm_lora"      # modèle fine-tuné avec LoRA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# 🔤 CHARGEMENT TOKENIZER & MODELE
# ================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    # Charger le fine-tuning LoRA
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ================================
# ▶️ GENERATION DE REPONSES
# ================================
def generate_response(user_input, max_length=200, temperature=0.8, top_p=0.9):
    """
    Génère une réponse en anglais depuis le modèle ARSLM LoRA.
    """
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
# 🖥 INTERFACE STREAMLIT
# ================================
st.set_page_config(page_title="ARSLM – MicroLLM SaaS", page_icon="🤖")

st.title("🤖 ARSLM – MicroLLM SaaS")
st.markdown("ARSLM is your intelligent, LoRA fine-tuned assistant (English).")

# Champ de texte utilisateur
user_input = st.text_input("Enter your message:", "")

if st.button("Send") and user_input:
    with st.spinner("Generating response..."):
        answer = generate_response(user_input)
    st.markdown(f"**ARSLM:** {answer}")