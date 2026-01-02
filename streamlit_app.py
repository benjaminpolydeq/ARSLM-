import streamlit as st
from arslm.profiles import PROFILES
from arslm.loader import load_model
from arslm.inference import generate

st.set_page_config(page_title="ARSLM Multi-Métiers", page_icon="🧠")

st.title("🧠 ARSLM – Assistant IA Multi-Métiers")

profile = st.sidebar.selectbox("Choisir le métier", PROFILES.keys())
profile_data = PROFILES[profile]

st.sidebar.info(profile_data["description"])

if profile_data["adapter"] in ["medical", "juridique", "police", "gouvernement"]:
    st.warning("⚠️ Informations générales – ne remplace pas un professionnel.")

model = load_model(profile_data["adapter"])

prompt = st.text_area("Votre question")

if st.button("Envoyer"):
    with st.spinner("ARSLM réfléchit..."):
        response = generate(model, prompt)
    st.success("Réponse")
    st.write(response)

