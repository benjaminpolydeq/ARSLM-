import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configuration de la page
st.set_page_config(
    page_title="ARSLM - Text Generation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .generated-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<div class="main-header">🧠 ARSLM Text Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Adaptive Reasoning Semantic Language Model</div>', unsafe_allow_html=True)

# Utiliser un modèle public accessible (pas de fichiers locaux)
# Vous pouvez changer pour votre modèle hébergé sur Hugging Face
MODEL_NAME = "gpt2"  # Remplacez par "votre-username/votre-modele" si vous avez uploadé votre modèle

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    """
    Charge le modèle depuis Hugging Face Hub
    """
    try:
        # Streamlit Cloud utilise CPU uniquement
        device = "cpu"
        
        # Chargement du tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Définir le pad_token si nécessaire
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Chargement du modèle (optimisé pour CPU)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(device)
        model.eval()
        
        # Création du pipeline
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1  # -1 pour CPU
        )
        
        return generator, tokenizer, None
        
    except Exception as e:
        return None, None, f"❌ Erreur lors du chargement: {str(e)}"

# Chargement du modèle avec barre de progression
with st.spinner('🔄 Chargement du modèle... (première fois peut prendre 1-2 minutes)'):
    generator, tokenizer, error = load_model_and_tokenizer()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if error:
        st.error(error)
    else:
        st.success(f"✅ Modèle chargé: {MODEL_NAME}")
    
    st.divider()
    
    st.subheader("📊 Paramètres")
    
    max_new_tokens = st.slider(
        'Tokens maximum',
        min_value=10,
        max_value=200,  # Limité pour Streamlit Cloud gratuit
        value=50,
        step=10,
        help="Nombre maximum de tokens à générer"
    )
    
    temperature = st.slider(
        'Température',
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Créativité (bas = conservateur, haut = créatif)"
    )
    
    top_k = st.slider(
        'Top-K',
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Nombre de tokens candidats"
    )
    
    top_p = st.slider(
        'Top-P',
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Probabilité cumulée"
    )
    
    st.divider()
    
    st.subheader("ℹ️ À propos")
    st.info("""
    **ARSLM** - Modèle de langage adaptatif pour:
    - Génération de texte
    - Compréhension contextuelle
    - Applications conversationnelles
    """)
    
    st.markdown("---")
    st.markdown("**Créé par:** Benjamin Amaad Kama")
    st.markdown("📧 benjokama@hotmail.fr")
    st.markdown("[GitHub](https://github.com/benjaminpolydeq/ARSLM)")

# Zone principale
if generator and tokenizer:
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["✍️ Génération", "🎯 Exemples", "📝 Historique"])
    
    with tab1:
        st.markdown("💡 **Entrez votre prompt ci-dessous et générez du texte intelligent !**")
        
        # Zone de texte
        prompt = st.text_area(
            'Votre prompt:',
            value="L'intelligence artificielle va transformer",
            height=100,
            help="Entrez le texte de départ"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            generate_btn = st.button('🚀 Générer', type="primary", use_container_width=True)
        
        with col2:
            clear_btn = st.button('🗑️ Effacer', use_container_width=True)
        
        # Génération
        if generate_btn and prompt:
            with st.spinner('✨ Génération en cours...'):
                try:
                    result = generator(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=1.2
                    )
                    
                    generated_text = result[0]['generated_text']
                    
                    st.success('✅ Génération terminée !')
                    st.subheader('📄 Texte généré')
                    st.markdown(f'<div class="generated-box">{generated_text}</div>', unsafe_allow_html=True)
                    
                    # Statistiques
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Caractères", len(generated_text))
                    with col_stat2:
                        st.metric("Mots", len(generated_text.split()))
                    with col_stat3:
                        st.metric("Tokens", len(tokenizer.encode(generated_text)))
                    
                    # Code copiable
                    st.code(generated_text, language=None)
                    
                    # Sauvegarder dans l'historique
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        'prompt': prompt,
                        'result': generated_text,
                        'params': {
                            'max_tokens': max_new_tokens,
                            'temperature': temperature,
                            'top_k': top_k,
                            'top_p': top_p
                        }
                    })
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    st.info("💡 Essayez de réduire le nombre de tokens ou de relancer")
        
        elif generate_btn:
            st.warning('⚠️ Veuillez entrer un prompt')
        
        if clear_btn:
            st.rerun()
    
    with tab2:
        st.subheader("🎯 Exemples de prompts")
        
        examples = {
            "🤖 Technologie": [
                "L'avenir de l'intelligence artificielle est",
                "Les robots du futur pourront",
                "La blockchain va révolutionner",
            ],
            "📚 Éducation": [
                "L'éducation en ligne permet",
                "Les étudiants de demain apprendront",
                "La technologie éducative transforme",
            ],
            "💼 Business": [
                "Les startups innovent en",
                "L'entrepreneuriat digital offre",
                "Le commerce électronique évolue vers",
            ],
            "🌍 Société": [
                "Le développement durable nécessite",
                "Les villes intelligentes vont",
                "La transformation numérique change",
            ]
        }
        
        for category, prompts in examples.items():
            with st.expander(category):
                for p in prompts:
                    if st.button(f"📝 {p}", key=p, use_container_width=True):
                        st.info(f"💡 Prompt sélectionné ! Allez dans l'onglet 'Génération' et collez: {p}")
    
    with tab3:
        st.subheader("📝 Historique des générations")
        
        if 'history' in st.session_state and st.session_state.history:
            for idx, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Génération #{len(st.session_state.history) - idx}"):
                    st.write("**Prompt:**", entry['prompt'])
                    st.write("**Résultat:**")
                    st.write(entry['result'])
                    st.json(entry['params'])
            
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("Aucune génération pour le moment. Commencez dans l'onglet 'Génération' !")

else:
    st.error("❌ Impossible de charger le modèle")
    
    with st.expander("📋 Informations de dépannage"):
        st.markdown("""
        ### Problèmes possibles:
        
        1. **Première utilisation**: Le téléchargement du modèle peut prendre 1-2 minutes
        2. **Connexion lente**: Vérifiez votre connexion Internet
        3. **Modèle indisponible**: Vérifiez que le modèle existe sur Hugging Face
        
        ### Solutions:
        
        - Rafraîchissez la page (F5)
        - Attendez quelques minutes
        - Contactez le support si le problème persiste
        """)
    
    st.info(f"Modèle utilisé: **{MODEL_NAME}**")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Propulsé par <strong>ARSLM</strong> | Créé avec ❤️ par Benjamin Amaad Kama</p>
        <p style='font-size: 0.9rem;'>
            <a href='https://github.com/benjaminpolydeq/ARSLM' target='_blank'>GitHub</a> | 
            <a href='mailto:benjokama@hotmail.fr'>Contact</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)