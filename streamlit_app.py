

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Define the path to the fine-tuned model
model_dir = './arslm_lora'

st.title('ARSLM Text Generation Demo')
st.write('Enter a prompt and let the ARSLM model generate text!')

@st.cache_resource
def load_model():
    if not os.path.exists(model_dir):
        st.error(f"Error: Model directory '{model_dir}' not found. Please ensure the fine-tuning process was completed.")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        # Ensure pad_token_id is set for generation
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        return generator, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

generator, tokenizer = load_model()

if generator:
    prompt = st.text_area('Your Prompt:', 'The future of artificial intelligence is')
    max_new_tokens = st.slider('Max New Tokens:', 10, 200, 50)
    temperature = st.slider('Temperature (creativity):', 0.1, 2.0, 0.7)
    top_k = st.slider('Top-K (diversity):', 0, 100, 50)

    if st.button('Generate Text'):
        if prompt:
            with st.spinner('Generating...'):
                generated_text = generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=temperature,
                    top_k=top_k
                )
                st.subheader('Generated Text:')
                st.write(generated_text[0]['generated_text'])
        else:
            st.warning('Please enter a prompt.')
else:
    st.warning('Model not loaded. Please ensure fine-tuning was successful and the model directory exists.')

