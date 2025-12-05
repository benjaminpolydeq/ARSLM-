import streamlit as st

st.set_page_config(page_title="ARSLM MVP", page_icon="🤖")

st.title("🤖 ARSLM – Version Test MVP")

st.write("Bienvenue sur le test Streamlit MVP !")

user_input = st.text_input("Pose une question :")

if st.button("Envoyer"):
    if user_input.strip() == "":
        st.warning("Merci d'entrer une question.")
    else:
        st.success("Réponse (test) :")
        st.write(f"ARSLM a reçu : **{user_input}**")
        st.write("⚙️ Le système fonctionne correctement.")
