import torch
from data_tokenize import DataTokenizer
from config import CONTEXT_SIZE , BATCH_SIZE
from model import BiGramModel

tokenizer = DataTokenizer("data.txt")
tokens = tokenizer.tokenize()
vocab_size = tokenizer.vocab_size

split = int(0.9*len(tokens))
train_data = tokens[:split]
test_data = tokens[split:]

def get_batches(mode):
    data = train_data if mode=="TRAIN" else test_data
    ix = torch.randint(len(data)-CONTEXT_SIZE,(BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_SIZE]for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1]for i in ix])
    return x,y

model = BiGramModel(vocab_size)
optim = torch.optim.AdamW(model.parameters(),lr=1e-3)

# xb,yb = get_batches("TRAIN")
print("Before Training \n")
print(tokenizer.decode(model.generate(torch.zeros((1,1),dtype=torch.long),max_new_token=400)[0].tolist()))

for steps in range(10000):
    xb , yb = get_batches("TRAIN")
    logits , loss = model(xb,yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    if(steps%100==0):
        print(loss)

print("After Training \n")
print(tokenizer.decode(model.generate(torch.zeros((1,1),dtype=torch.long),max_new_token=400)[0].tolist()))