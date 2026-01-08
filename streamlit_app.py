import streamlit as st
import sys

# Patch pour compatibilité Python 3.13 avec transformers
try:
    from transformers.generation import GenerationMixin
except ImportError:
    try:
        from transformers.generation.utils import GenerationMixin
        sys.modules['transformers.generation'].GenerationMixin = GenerationMixin
    except Exception as e:
        st.error(f"⚠️ Problème de compatibilité: {e}")
        st.info("Essayez: pip install transformers==4.44.2 --upgrade")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Configuration de la page
st.set_page_config(
    page_title="ARSLM - Text Generation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
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
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    .generated-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<div class="main-header">🧠 ARSLM Text Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Adaptive Reasoning Semantic Language Model</div>', unsafe_allow_html=True)

# Chemin du modèle
MODEL_DIR = './arslm_lora'

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """
    Charge le modèle et le tokenizer avec gestion d'erreurs
    """
    if not os.path.exists(MODEL_DIR):
        return None, None, f"❌ Le dossier du modèle '{MODEL_DIR}' n'existe pas."
    
    try:
        with st.spinner('🔄 Chargement du modèle...'):
            # Détection du device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Chargement du tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True
            )
            
            # Définir le pad_token si nécessaire
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Chargement du modèle
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            model.to(device)
            model.eval()
            
            # Création du pipeline
            generator = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1
            )
            
            return generator, tokenizer, None
            
    except Exception as e:
        return None, None, f"❌ Erreur lors du chargement: {str(e)}"

# Chargement du modèle
generator, tokenizer, error = load_model_and_tokenizer()

# Sidebar avec informations
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if error:
        st.error(error)
        st.info("💡 **Solution:** Assurez-vous que le modèle fine-tuné existe dans le dossier `./arslm_lora`")
    else:
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.success(f"✅ Modèle chargé sur {device}")
    
    st.divider()
    
    st.subheader("📊 Paramètres de génération")
    
    max_new_tokens = st.slider(
        'Tokens maximum',
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Nombre maximum de tokens à générer"
    )
    
    temperature = st.slider(
        'Température',
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Contrôle la créativité (bas = conservateur, haut = créatif)"
    )
    
    top_k = st.slider(
        'Top-K',
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Nombre de tokens candidats à considérer"
    )
    
    top_p = st.slider(
        'Top-P (nucleus)',
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Probabilité cumulée pour la sélection des tokens"
    )
    
    num_sequences = st.number_input(
        'Nombre de variantes',
        min_value=1,
        max_value=5,
        value=1,
        help="Nombre de textes différents à générer"
    )
    
    st.divider()
    
    st.subheader("ℹ️ À propos")
    st.info("""
    **ARSLM** est un modèle de langage adaptatif conçu pour:
    - Génération de texte intelligent
    - Compréhension contextuelle
    - Adaptation aux flux dynamiques
    """)
    
    st.markdown("---")
    st.markdown("**Créé par:** Benjamin Amaad Kama")
    st.markdown("📧 benjokama@hotmail.fr")

# Zone principale
if generator and tokenizer:
    st.markdown('<div class="info-box">💡 Entrez votre prompt ci-dessous et cliquez sur "Générer" pour voir la magie opérer!</div>', unsafe_allow_html=True)
    
    # Onglets pour différents modes
    tab1, tab2, tab3 = st.tabs(["✍️ Génération libre", "🎯 Prompts prédéfinis", "📝 Historique"])
    
    with tab1:
        # Zone de saisie du prompt
        prompt = st.text_area(
            'Votre prompt:',
            value='Le futur de l\'intelligence artificielle est',
            height=150,
            help="Entrez le texte de départ pour la génération"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            generate_button = st.button('🚀 Générer le texte', type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button('🗑️ Effacer', use_container_width=True)
        
        with col3:
            if st.button('💾 Sauvegarder', use_container_width=True):
                st.info("Fonctionnalité à venir!")
        
        # Génération
        if generate_button and prompt:
            with st.spinner('✨ Génération en cours...'):
                try:
                    results = generator(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=num_sequences,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=1.2
                    )
                    
                    st.success(f'✅ Génération terminée! ({num_sequences} variante{"s" if num_sequences > 1 else ""})')
                    
                    # Affichage des résultats
                    for idx, result in enumerate(results):
                        if num_sequences > 1:
                            st.subheader(f'📄 Variante {idx + 1}')
                        else:
                            st.subheader('📄 Texte généré')
                        
                        generated_text = result['generated_text']
                        st.markdown(f'<div class="generated-box">{generated_text}</div>', unsafe_allow_html=True)
                        
                        # Statistiques
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Longueur totale", f"{len(generated_text)} caractères")
                        with col_stat2:
                            words = len(generated_text.split())
                            st.metric("Mots", words)
                        with col_stat3:
                            tokens = len(tokenizer.encode(generated_text))
                            st.metric("Tokens", tokens)
                        
                        # Bouton de copie
                        st.code(generated_text, language=None)
                        
                        if idx < len(results) - 1:
                            st.divider()
                    
                    # Sauvegarder dans l'historique
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        'prompt': prompt,
                        'results': results,
                        'params': {
                            'max_tokens': max_new_tokens,
                            'temperature': temperature,
                            'top_k': top_k,
                            'top_p': top_p
                        }
                    })
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération: {str(e)}")
        
        elif generate_button and not prompt:
            st.warning('⚠️ Veuillez entrer un prompt avant de générer.')
        
        if clear_button:
            st.rerun()
    
    with tab2:
        st.subheader("🎯 Prompts suggérés")
        
        prompts_suggestions = {
            "🤖 IA et Technologie": [
                "L'intelligence artificielle transformera l'Afrique en",
                "Les avancées en machine learning permettent",
                "Dans le futur, les robots pourront",
            ],
            "📚 Éducation": [
                "L'éducation en ligne révolutionne",
                "Les technologies éducatives aident les étudiants à",
                "L'avenir de l'apprentissage sera",
            ],
            "💼 Business": [
                "Les startups africaines innovent en",
                "Le commerce électronique en Afrique",
                "Les opportunités d'entrepreneuriat dans",
            ],
            "🌍 Société": [
                "Le développement durable nécessite",
                "Les défis de l'urbanisation en Afrique",
                "La transformation digitale change",
            ]
        }
        
        for category, prompts in prompts_suggestions.items():
            with st.expander(category):
                for prompt_text in prompts:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"• {prompt_text}")
                    with col2:
                        if st.button("Utiliser", key=prompt_text):
                            st.session_state.selected_prompt = prompt_text
                            st.switch_page
    
    with tab3:
        st.subheader("📝 Historique des générations")
        
        if 'history' in st.session_state and st.session_state.history:
            for idx, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Génération #{len(st.session_state.history) - idx} - {entry['prompt'][:50]}..."):
                    st.write("**Prompt:**", entry['prompt'])
                    st.write("**Paramètres:**")
                    st.json(entry['params'])
                    st.write("**Résultat:**")
                    st.write(entry['results'][0]['generated_text'])
            
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("Aucune génération dans l'historique. Commencez par générer du texte!")

else:
    st.error("❌ Le modèle n'a pas pu être chargé.")
    
    with st.expander("📋 Instructions de dépannage"):
        st.markdown("""
        ### Étapes de résolution:
        
        1. **Vérifiez le chemin du modèle:**
           ```bash
           ls -la ./arslm_lora
           ```
        
        2. **Assurez-vous que le fine-tuning est terminé:**
           - Le dossier doit contenir `config.json`, `pytorch_model.bin`, etc.
        
        3. **Vérifiez les permissions:**
           ```bash
           chmod -R 755 ./arslm_lora
           ```
        
        4. **Réinstallez les dépendances:**
           ```bash
           pip install -r requirements.txt --upgrade
           ```
        
        5. **Consultez les logs pour plus d'informations**
        """)
    
    st.info("""
    💡 **Besoin d'aide?**
    - Email: benjokama@hotmail.fr
    - GitHub: github.com/benjaminpolydeq/ARSLM
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Propulsé par <strong>ARSLM</strong> | Créé avec ❤️ par Benjamin Amaad Kama</p>
        <p style='font-size: 0.9rem;'>© 2025 - Tous droits réservés</p>
    </div>
    """,
    unsafe_allow_html=True
)