"""
arslm.py - Implémentation du modèle ARSLM minimaliste

Ce fichier contient :
- ARSCell : cellule adaptative inspirée de BenPolySeq.
- LocalCausalAttention : attention causale locale pour capturer le contexte.
- ARSLM : modèle complet combinant embedding, ARSCell, attention et sortie vocabulaire.

Usage :
import torch
from arslm import ARSLM

model = ARSLM(vocab_size=2000)
input_ids = torch.randint(0,2000,(1,10))
logits = model(input_ids)  # (batch, seq_len, vocab_size)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ARSCell(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.candidate_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + emb_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim*2 + emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
        self.res_proj = nn.Linear(emb_dim, hidden_dim)

    def forward(self, h_prev2, h_prev1, x_embed):
        ctx = torch.cat([h_prev1,h_prev2,x_embed],dim=-1)
        candidate = self.candidate_mlp(ctx)
        gate = self.gate_net(ctx).squeeze(-1)
        residual = self.res_proj(x_embed)
        h_t = h_prev1 + gate.unsqueeze(-1)*candidate + 0.1*residual
        return F.layer_norm(h_t,(self.hidden_dim,))

class LocalCausalAttention(nn.Module):
    def __init__(self, hidden_dim, window=8):
        super().__init__()
        self.window = window
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, states):
        B,T,H = states.size()
        outs=[]
        for t in range(T):
            start = max(0, t-self.window)
            ctx = states[:,start:t+1,:]
            q = self.query(states[:,t:t+1,:])
            k = self.key(ctx)
            v = self.value(ctx)
            scores = torch.matmul(q,k.transpose(-1,-2))/(H**0.5)
            w = F.softmax(scores,dim=-1)
            outs.append(torch.matmul(w,v))
        return torch.cat(outs,dim=1)

class ARSLM(nn.Module):
    def __init__(self,vocab_size=2000,emb_dim=64,hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,emb_dim)
        self.cell = ARSCell(emb_dim,hidden_dim)
        self.attn = LocalCausalAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim,vocab_size)

    def forward(self,x):
        B,T = x.size()
        e = self.embedding(x)
        h_prev2 = torch.zeros(B,self.cell.hidden_dim).to(x.device)
        h_prev1 = torch.zeros(B,self.cell.hidden_dim).to(x.device)
        states=[]
        for t in range(T):
            h_t = self.cell(h_prev2,h_prev1,e[:,t,:])
            states.append(h_t)
            h_prev2,h_prev1 = h_prev1,h_t
        states = torch.stack(states,dim=1)
        attn = self.attn(states)
        return self.fc(attn)