import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "distilgpt2"          # ou ton modèle de base
LORA_PATH = "./arslm_lora"          # dossier LoRA fine-tuné
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# PROMPT SYSTÈME (FREEZE)
# =========================
SYSTEM_PROMPT = """Tu es ARSLM, un assistant intelligent, professionnel et fiable.
Tu réponds en français, de manière claire, structurée et utile.
Tu ne répètes jamais inutilement les mots.
Tu expliques les concepts de façon pédagogique.
Si une question est ambiguë, tu demandes une clarification.
Réponds toujours de manière naturelle et cohérente.
"""

# =========================
# CHARGEMENT MODÈLE
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(DEVICE)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# =========================
# UI STREAMLIT
# =========================
st.title("🧠 ARSLM – MicroLLM SaaS")
st.write("Le modèle ARSLM est prêt à être testé.")

user_input = st.text_area("Entrez un texte pour ARSLM :", height=120)

if st.button("Générer la réponse"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer une question.")
    else:
        # PROMPT FINAL
        prompt = f"""
{SYSTEM_PROMPT}

Question : {user_input}
Réponse :
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Réponse :")[-1].strip()

        st.subheader("Réponse ARSLM :")
        st.write(response)