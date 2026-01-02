import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================================
# 🔹 Config
# ================================
MODEL_PATH = "./model_checkpoint"     # chemin vers ton modèle fine-tuné (ou base si LoRA présent)
TOKENIZER_PATH = "./tokenizer_checkpoint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# 🔹 Charger modèle et tokenizer
# ================================
@st.cache_resource(show_spinner=True)
def load_model():
    # Charger tokenizer (tombe sur TOKENIZER_PATH si renseigné)
    tokenizer_path = TOKENIZER_PATH if TOKENIZER_PATH else MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # Essayer de charger un modèle PEFT (LoRA) si le checkpoint contient des adaptateurs,
    # sinon charger un modèle causal standard.
    try:
        # Charger le modèle de base
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Essayer d'envelopper avec PeftModel (si MODEL_PATH contient des poids LoRA)
        model = PeftModel.from_pretrained(base_model, MODEL_PATH, device_map="auto" if torch.cuda.is_available() else None)
    except Exception:
        # Fallback : charger directement le modèle causal si Peft échoue
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto" if torch.cuda.is_available() else None)

    # Si device_map="auto" a été utilisé, le modèle est probablement déjà sur le bon device.
    # Sinon on force le to(DEVICE).
    try:
        model.to(DEVICE)
    except Exception:
        pass

    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# ================================
# 🔹 Génération de réponse
# ================================
def generate_response(user_input, max_length=200, temperature=0.8, top_p=0.9):
    prompt = f"You are ARSLM, an intelligent and friendly assistant that speaks English.\nUser: {user_input}\nARSLM:"
    inputs = tokenizer(prompt, return_tensors="pt")
    # déplacer les tenseurs vers l'appareil si nécessaire
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("ARSLM:")[-1].strip()
    return response

# ================================
# 🔹 Streamlit Interface
# ================================
st.set_page_config(page_title="ARSLM – MicroLLM SaaS", page_icon="🤖")
st.title("🤖 ARSLM - MicroLLM SaaS")
st.write("ARSLM est prêt à discuter en anglais !")

user_input = st.text_input("💬 Pose une question à ARSLM :")

if st.button("Envoyer") and user_input.strip() != "":
    with st.spinner("ARSLM réfléchit..."):
        answer = generate_response(user_input)
        st.markdown(f"**ARSLM:** {answer}")
