# arslm_fixes.py
# Module de corrections pour ARSLM.ipynb — importez-le depuis votre notebook
# Usage rapide dans le notebook :
#   !pip install transformers --quiet
#   from arslm_fixes import AdvancedTokenizer, ARSCell, ARSLM, collate_batch, train_demo
#   model, tokenizer = train_demo()

import math
import random
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utiliser la version fast si disponible
try:
    from transformers import BertTokenizerFast as HFTokenizer
except Exception:
    from transformers import BertTokenizer as HFTokenizer


class AdvancedTokenizer:
    """
    Wrapper simple autour d'un tokenizer Hugging Face.
    - encode(text) -> List[int] (sans special tokens)
    - decode(ids) -> str (skip special tokens)
    - len(tokenizer) renvoie la taille du vocab
    - expose les ids pour pad/bos/eos
    """
    def __init__(self, pretrained_name: str = "bert-base-uncased"):
        # Charger le tokenizer (cache HF normalement)
        self.tokenizer = HFTokenizer.from_pretrained(pretrained_name, use_fast=True)
        # récupère le dict token->id
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {i: w for w, i in self.stoi.items()}

        # Spécial tokens et ids (compatibilité)
        self.pad_token = getattr(self.tokenizer, "pad_token", "[PAD]")
        self.unk_token = getattr(self.tokenizer, "unk_token", "[UNK]")
        # On utilise CLS comme BOS et SEP comme EOS par convention BERT-like
        self.bos_token = getattr(self.tokenizer, "cls_token", "[CLS]")
        self.eos_token = getattr(self.tokenizer, "sep_token", "[SEP]")

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.unk_token_id = getattr(self.tokenizer, "unk_token_id", 100)
        self.bos_token_id = getattr(self.tokenizer, "cls_token_id", 101)
        self.eos_token_id = getattr(self.tokenizer, "sep_token_id", 102)

    def encode(self, text: str) -> List[int]:
        # retourne ids sans ajouter les special tokens (on ajoutera BOS/EOS si besoin)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def __len__(self):
        # Vocab size
        return len(self.stoi)


class ARSCell(nn.Module):
    """
    Cellule ARS : calcule h_t à partir de h_{t-2}, h_{t-1} et x_embed.
    h_t = h_{t-1} + gate * candidate + small_residual
    """
    def __init__(self, emb_dim: int, hidden_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # candidate MLP: input dim = h_prev1 + h_prev2 + x_embed
        self.candidate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + emb_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # gate network -> scalar in (0,1)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.res_proj = nn.Linear(emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, h_prev2: torch.Tensor, h_prev1: torch.Tensor, x_embed: torch.Tensor):
        # expected shapes:
        #   h_prev2, h_prev1: (batch, hidden_dim)
        #   x_embed: (batch, emb_dim)
        ctx = torch.cat([h_prev1, h_prev2, x_embed], dim=-1)
        candidate = self.candidate_mlp(ctx)                # (batch, hidden_dim)
        gate = self.gate_net(ctx).squeeze(-1)              # (batch,)
        residual = self.res_proj(x_embed)
        # adaptative update
        h_t = h_prev1 + gate.unsqueeze(-1) * candidate + 0.1 * residual
        h_t = self.dropout(h_t)
        # stable layer norm
        h_t = F.layer_norm(h_t, (self.hidden_dim,))
        return h_t, gate


class ARSLM(nn.Module):
    """
    ARSLM: embedding -> stacked ARSCell(s) -> additive attention -> head
    Designed to be simple and robust for small demos.
    """
    def __init__(self, tokenizer: AdvancedTokenizer, emb_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.emb = nn.Embedding(self.vocab_size, emb_dim, padding_idx=tokenizer.pad_token_id)
        # cells: first layer input is emb_dim, others receive hidden_dim
        self.cells = nn.ModuleList([
            ARSCell(emb_dim if i == 0 else hidden_dim, hidden_dim, dropout_prob=0.1)
            for i in range(num_layers)
        ])
        # attention scoring (maps hidden_dim -> score)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.head = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: (batch, seq_len)
        returns logits: (batch, seq_len, vocab) and gates from last layer (batch, seq_len)
        """
        bsz, seq_len = input_ids.shape
        emb = self.emb(input_ids)            # (b, seq, emb_dim)
        device = emb.device
        hidden_dim = self.hidden_dim

        # initialize previous states per layer
        h_prev2_list = [torch.zeros(bsz, hidden_dim, device=device) for _ in range(self.num_layers)]
        h_prev1_list = [torch.zeros(bsz, hidden_dim, device=device) for _ in range(self.num_layers)]

        all_last_layer_hidden_states = []  # store history for attention
        logits_list = []
        gates_list = []

        for t in range(seq_len):
            x_t = emb[:, t, :]   # (b, emb_dim)
            h_t_input = x_t

            current_layer_hidden_states = []

            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h_prev2 = h_prev2_list[layer]
                h_prev1 = h_prev1_list[layer]

                if layer > 0:
                    h_t_input = current_layer_hidden_states[-1]

                h_t, gate = cell(h_prev2, h_prev1, h_t_input)
                current_layer_hidden_states.append(h_t)

                # update for next time-step
                h_prev2_list[layer] = h_prev1
                h_prev1_list[layer] = h_t

                if layer == self.num_layers - 1:
                    gates_list.append(gate.unsqueeze(1))  # (b,1)

            last_layer_h_t = current_layer_hidden_states[-1]  # (b, hidden_dim)
            all_last_layer_hidden_states.append(last_layer_h_t.unsqueeze(1))  # keep history

            # attention over history (causal), last_layer_history shape: (b, t+1, hidden_dim)
            last_layer_history = torch.cat(all_last_layer_hidden_states, dim=1)  # (b, t+1, hidden_dim)
            # score each history element
            scores = self.attention(last_layer_history).squeeze(-1)  # (b, t+1)
            attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)    # (b, t+1, 1)
            context = torch.sum(attn_weights * last_layer_history, dim=1)  # (b, hidden_dim)

            attended_h_t = last_layer_h_t + context
            logit = self.head(attended_h_t)   # (b, vocab)
            logits_list.append(logit.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1) if logits_list else torch.zeros(bsz, 0, self.vocab_size, device=device)
        gates = torch.cat(gates_list, dim=1) if gates_list else torch.zeros(bsz, 0, device=device)
        return logits, gates

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 20, temperature: float = 1.0):
        """
        idx: (batch, seq_len) initial context
        returns: list of lists (token ids) expanded
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.eval()
        bsz, seq_len = idx.shape
        device = next(self.parameters()).device
        hidden_dim = self.hidden_dim

        # init states
        h_prev2_list = [torch.zeros(bsz, hidden_dim, device=device) for _ in range(self.num_layers)]
        h_prev1_list = [torch.zeros(bsz, hidden_dim, device=device) for _ in range(self.num_layers)]
        all_last_layer_hidden_states = []

        # process initial context to set starting hidden states
        emb = self.emb(idx)  # (b, seq_len, emb_dim)
        for t in range(seq_len):
            x_t = emb[:, t, :]
            h_t_input = x_t
            current_layer_hidden_states = []
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h_prev2 = h_prev2_list[layer]
                h_prev1 = h_prev1_list[layer]
                if layer > 0:
                    h_t_input = current_layer_hidden_states[-1]
                h_t, _ = cell(h_prev2, h_prev1, h_t_input)
                current_layer_hidden_states.append(h_t)
                h_prev2_list[layer] = h_prev1
                h_prev1_list[layer] = h_t
                if layer == self.num_layers - 1:
                    all_last_layer_hidden_states.append(h_t.unsqueeze(1))

        out_ids = [list(row.tolist()) for row in idx.cpu()]

        for _ in range(max_new_tokens):
            # embedding of last token for each sequence in the batch
            last_token_ids = torch.tensor([ids[-1] for ids in out_ids], dtype=torch.long, device=device).unsqueeze(1)
            x_embed = self.emb(last_token_ids).squeeze(1)

            h_t_input = x_embed
            current_layer_hidden_states = []
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h_prev2 = h_prev2_list[layer]
                h_prev1 = h_prev1_list[layer]
                if layer > 0:
                    h_t_input = current_layer_hidden_states[-1]
                h_t, _ = cell(h_prev2, h_prev1, h_t_input)
                current_layer_hidden_states.append(h_t)
                h_prev2_list[layer] = h_prev1
                h_prev1_list[layer] = h_t
                if layer == self.num_layers - 1:
                    all_last_layer_hidden_states.append(h_t.unsqueeze(1))

            last_layer_h_t = current_layer_hidden_states[-1]
            last_layer_history = torch.cat(all_last_layer_hidden_states, dim=1)  # (b, seq, hidden_dim)
            scores = self.attention(last_layer_history).squeeze(-1)
            attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
            context = torch.sum(attn_weights * last_layer_history, dim=1)
            attended_h_t = last_layer_h_t + context

            logits = self.head(attended_h_t) / float(temperature)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (b,)

            for i in range(bsz):
                out_ids[i].append(int(next_token[i].item()))

        return out_ids


# Utilities


def collate_batch(tokenizer: AdvancedTokenizer, texts: List[str], device: torch.device):
    """
    Use HF tokenizer batching to produce padded input_ids tensor.
    Returns tensor on device.
    """
    enc = tokenizer.tokenizer(
        texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    return input_ids


def train_demo(device: torch.device = None, n_epochs: int = 200, seed: int = 0):
    """
    Small demo training using the corrected classes.
    Returns model, tokenizer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    texts = [
        "hello world this is ars",
        "the system adapts to its history",
        "benpolyseq demonstrates adaptive sequences",
        "ars can inspire new network protocols",
        "self optimizing systems are possible",
    ]

    tokenizer = AdvancedTokenizer()
    model = ARSLM(tokenizer, emb_dim=64, hidden_dim=128, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    batch = collate_batch(tokenizer, texts, device)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    for epoch in range(n_epochs):
        model.train()
        logits, gates = model(inputs)  # logits: (b, seq, vocab)
        b, seq, v = logits.shape
        loss = loss_fn(logits.view(b * seq, v), targets.reshape(b * seq))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — loss: {loss.item():.4f}")

    # Demo generation
    context = "hello world"
    idx = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(context)], dtype=torch.long, device=device)
    out_ids = model.generate(idx, max_new_tokens=15, temperature=1.0)[0]
    print("=== Generated ===")
    print(tokenizer.decode(out_ids))

    return model, tokenizer
