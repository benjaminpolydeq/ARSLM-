import streamlit as st
import random

# --- CONFIG ---
st.set_page_config(page_title="MicroLLM Démo", layout="wide")
st.title("🚀 MicroLLM Studio - Démo ARSLM")

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state["history"] = []


# --- FONCTION DE GENERATION FACTICE ---
def generate_response(prompt: str) -> str:
    responses = [
        "Bonjour ! Comment puis-je vous aider ?",
        "Merci pour votre message !",
        "Notre équipe reviendra vers vous rapidement.",
        "Pouvez-vous préciser votre demande ?",
        "Nous avons bien reçu votre requête."
    ]
    return random.choice(responses)


# --- FORMULAIRE CHAT ---
with st.form("chat_form"):
    user_input = st.text_input("💬 Votre message", key="chat_input")
    submit = st.form_submit_button("Envoyer")

if submit and user_input:
    user_msg = user_input.strip()
    if user_msg:
        bot_response = generate_response(user_msg)
        st.session_state["history"].append(("Vous", user_msg))
        st.session_state["history"].append(("IA", bot_response))


# --- AFFICHAGE DE L'HISTORIQUE DANS UN CONTAINER STABLE ---
chat_container = st.container()

with chat_container:
    for speaker, msg in st.session_state["history"]:
        st.markdown(f"**{speaker} :** {msg}")