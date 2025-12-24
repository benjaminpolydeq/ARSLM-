# streamlit_app.py
import streamlit as st
from datetime import datetime

# ⚠️ Assurez-vous que ARSLM est dans arslm/arslm.py
from arslm.arslm import ARSLM  

st.set_page_config(page_title="ARSLM Chat", layout="wide")

st.title("🧠 ARSLM Chatbot")
st.write("Chat interactif avec ARSLM (Adaptive Reasoning Semantic Language Model)")

# ---------------------------
# 1️⃣ Initialisation session
# ---------------------------
if "arslm_session" not in st.session_state:
    try:
        # Initialisation ARSLM custom CPU
        st.session_state.arslm_session = ARSLM(device="cpu", custom_model=True)
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation d'ARSLM: {e}")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# 2️⃣ Reset bouton
# ---------------------------
if st.button("🔄 Reset Conversation"):
    if "arslm_session" in st.session_state:
        st.session_state.arslm_session.clear_history()
    st.session_state.history = []
    st.experimental_rerun()

# ---------------------------
# 3️⃣ Affichage historique
# ---------------------------
if st.session_state.history:
    st.subheader("💬 Conversation")
    for exchange in st.session_state.history:
        st.markdown(f"**User:** {exchange['user']}")
        st.markdown(f"**Assistant:** {exchange['assistant']}")
        st.markdown("---")

# ---------------------------
# 4️⃣ Input utilisateur
# ---------------------------
user_input = st.text_input("Entrez votre message:", key="input")

if st.button("Envoyer") and user_input.strip():
    if "arslm_session" in st.session_state:
        with st.spinner("📝 Génération en cours..."):
            try:
                # Générer réponse
                assistant_response = st.session_state.arslm_session.generate(
                    prompt=user_input,
                    max_length=150,
                    temperature=0.7,
                    include_context=True
                )

                # Mettre à jour historique
                st.session_state.history.append({
                    "user": user_input,
                    "assistant": assistant_response,
                    "timestamp": datetime.now().isoformat()
                })

                # Afficher la dernière réponse
                st.markdown(f"**Assistant:** {assistant_response}")

                # Vider champ input
                st.session_state.input = ""
            except Exception as e:
                st.error(f"Erreur lors de la génération de réponse: {e}")
    else:
        st.error("La session ARSLM n'a pas été initialisée correctement.")

# ---------------------------
# 5️⃣ Sidebar info
# ---------------------------
st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"Nombre d'échanges: {len(st.session_state.history)}")