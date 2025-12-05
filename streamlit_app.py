import streamlit as st
import random

st.set_page_config(page_title="MicroLLM Démo", layout="wide")
st.title("🚀 MicroLLM Studio - Démo ARSLM")

# Historique du chat
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Fonction de génération factice
def generate_response(prompt):
    responses = [
        "Bonjour ! Comment puis-je vous aider ?",
        "Merci pour votre message !",
        "Notre équipe reviendra vers vous rapidement.",
        "Pouvez-vous préciser votre demande ?",
        "Nous avons bien reçu votre requête."
    ]
    return random.choice(responses)

# Formulaire de chat
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Votre message")
    submit = st.form_submit_button("Envoyer")

if submit and user_input:
    response = generate_response(user_input)
    st.session_state['history'].append(("Vous", user_input))
    st.session_state['history'].append(("IA", response))

# Affichage de l'historique
for speaker, msg in st.session_state['history']:
    if speaker == "Vous":
        st.markdown(f"**{speaker}:** {msg}")
    else:
        st.markdown(f"**{speaker}:** {msg}")
