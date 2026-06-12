import torch
from data_tokenize import DataTokenizer
from config import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    DATA_PATH,
    DEVICE,
    EVAL_INTERVAL,
    EVAL_ITERS,
    GENERATE_TOKENS,
    LEARNING_RATE,
    MAX_ITER,
    N_EMBED,
    TRAIN_SPLIT,
)
from model import BiGramModel

tokenizer = DataTokenizer(DATA_PATH)
tokens = tokenizer.tokenize()
vocab_size = tokenizer.vocab_size

split = int(TRAIN_SPLIT*len(tokens))
train_data = tokens[:split]
test_data = tokens[split:]

def get_batches(mode):
    data = train_data if mode.lower()=="train" else test_data
    ix = torch.randint(len(data)-CONTEXT_SIZE,(BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_SIZE]for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1]for i in ix])
    return x.to(DEVICE),y.to(DEVICE)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batches(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

'''Bi-GramModel
 has no context, prediction based completly on the previous charecter.
'''
model = BiGramModel(vocab_size).to(DEVICE)
optim = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)

# xb,yb = get_batches("TRAIN")
print("Before Training \n")
print(tokenizer.decode(model.generate(torch.zeros((1,1),dtype=torch.long,device=DEVICE),max_new_token=GENERATE_TOKENS)[0].tolist()))

for iter in range(MAX_ITER):
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITER - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb , yb = get_batches("TRAIN")
    logits , loss = model(xb,yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

print("After Training \n")
print(tokenizer.decode(model.generate(torch.zeros((1,1),dtype=torch.long,device=DEVICE),max_new_token=GENERATE_TOKENS)[0].tolist()))


'''adding context to tokens by averaging

'''
x = torch.randn(BATCH_SIZE,CONTEXT_SIZE,N_EMBED)
x_bow = torch.zeros(BATCH_SIZE,CONTEXT_SIZE,N_EMBED)

for b in range(BATCH_SIZE):
    for t in range(CONTEXT_SIZE):
        x_prev = x[b,:t+1]
        x_bow[b,t] = torch.mean(x_prev,0)

print(x_bow)
