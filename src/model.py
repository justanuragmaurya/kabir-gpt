import torch
import torch.nn as nn
from torch.nn import functional as F
from config import N_EMBED , CONTEXT_SIZE

class BiGramModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,N_EMBED)
        self.pos_embedding_table = nn.Embedding(block_size,N_EMBED)
        self.lm_head = nn.Linear(N_EMBED,vocab_size)
    
    def forward(self,idx,targets=None):
        tok_embedings  = self.token_embedding_table(idx)
        pos_embedding = self.pos_embedding_table(torch.arange(CONTEXT_SIZE))
        x = tok_embedings+pos_embedding
        logits = self.lm_head(x)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits,loss
    
    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            logits , loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx