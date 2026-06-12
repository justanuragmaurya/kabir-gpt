import torch
import torch.nn as nn
from torch.nn import functional as F
from config import N_EMBED , CONTEXT_SIZE 

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED,head_size,bias=False)
        self.query = nn.Linear(N_EMBED,head_size,bias=False)
        self.value = nn.Linear(N_EMBED,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(CONTEXT_SIZE,CONTEXT_SIZE)))
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q@ k.transpose(-2,-1) * C **-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.projection = nn.Linear(N_EMBED,N_EMBED)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.projection(out)
        return out


class FeedFwd(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed),
            nn.ReLU(),
            nn.Linear(n_embed,n_embed),
        )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.self_attn = MultiHeadAttention(n_head,head_size)
        self.feed_fwd = FeedFwd(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.feed_fwd(self.ln2(x))
        return x

class BiGramModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,N_EMBED)
        self.pos_embedding_table = nn.Embedding(CONTEXT_SIZE,N_EMBED)
        # # self.attn_head = Head(N_EMBED)
        # self.attn_head = MultiHeadAttention(4,N_EMBED//4)
        # self.feed_fwd = FeedFwd(N_EMBED)
        self.blocks = nn.Sequential(
            Block(N_EMBED,n_head=4),
            Block(N_EMBED,n_head=4),
            Block(N_EMBED,n_head=4),
            nn.LayerNorm(N_EMBED)
        )
        self.lm_head = nn.Linear(N_EMBED,vocab_size)
    
    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_embedings  = self.token_embedding_table(idx)
        pos_embedding = self.pos_embedding_table(torch.arange(T))
        x = tok_embedings+pos_embedding
        # x = self.attn_head(x)
        # x = self.feed_fwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits,loss
    
    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -CONTEXT_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx