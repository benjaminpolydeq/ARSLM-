import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# 1) ARSLM Cellule (très simple)
# ============================
class ARSCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.candidate_mlp = nn.Sequential(
            nn.Linear(hidden_size*2 + input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size*2 + input_size, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x_embed, h_prev, h_prev2):
        context = torch.cat([h_prev, h_prev2, x_embed], dim=-1)
        candidate = self.candidate_mlp(context)
        gate = self.gate_net(context)
        residual = 0.1 * x_embed
        h_next = h_prev + gate * candidate + residual
        h_next = self.layer_norm(h_next)
        return h_next

# ============================
# 2) Modèle ARSLM léger
# ============================
class ARSLMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_size=64, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList([ARSCell(hidden_size, emb_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.emb(input_ids)
        h_prev = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        h_prev2 = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = x[:,t,:]
            for l in range(self.num_layers):
                h_next = self.layers[l](x_t, h_prev[l], h_prev2[l])
                h_prev2[l] = h_prev[l]
                h_prev[l] = h_next
                x_t = h_next
            logits = self.head(h_prev[-1])
            outputs.append(logits.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# ============================
# 3) Wrapper MicroLLM pour démo
# ============================
class MicroLLMDemo:
    def __init__(self):
        # Vocabulaire ultra simple
        self.vocab = {"bonjour":0, "merci":1, "client":2, "achat":3, "service":4, "STOP":5}
        self.inv_vocab = {i:w for w,i in self.vocab.items()}
        self.model = ARSLMModel(vocab_size=len(self.vocab))
        self.device = "cpu"
        self.model.to(self.device)

    def encode(self, text):
        tokens = [self.vocab.get(w.lower(),0) for w in text.split()]
        return torch.tensor([tokens], dtype=torch.long)

    def decode(self, toks):
        return " ".join(self.inv_vocab.get(int(t), "<unk>") for t in toks)

    @torch.no_grad()
    def generate(self, prompt):
        x = self.encode(prompt).to(self.device)
        logits = self.model(x)
        toks = torch.argmax(logits, dim=-1)[0]
        return self.decode(toks)

# ============================
# 4) Streamlit interface
# ============================
st.set_page_config(page_title="MicroLLM Démo", layout="wide")
st.title("🚀 MicroLLM Studio - Démo ARSLM")

# Initialiser le MicroLLM de démonstration
if 'llm' not in st.session_state:
    st.session_state['llm'] = MicroLLMDemo()
if 'history' not in st.session_state:
    st.session_state['history'] = []

llm = st.session_state['llm']

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Votre message")
    submit = st.form_submit_button("Envoyer")

if submit and user_input:
    response = llm.generate(user_input)
    st.session_state['history'].append(("Vous", user_input))
    st.session_state['history'].append(("IA", response))

# Afficher l'historique du chat
for speaker, msg in st.session_state['history']:
    if speaker == "Vous":
        st.markdown(f"**{speaker}:** {msg}")
    else:
        st.markdown(f"**{speaker}:** {msg}")
