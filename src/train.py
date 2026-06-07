import torch
from data_tokenize import DataTokenizer
from config import CONTEXT_SIZE , BATCH_SIZE

tokenizer = DataTokenizer("data.txt")
tokens = tokenizer.tokenize()

split = int(0.9*len(tokens))
train_data = tokens[:split]
test_data = tokens[split:]

def get_batches(mode):
    data = train_data if mode=="TRAIN" else test_data
    ix = torch.randint(len(data)-CONTEXT_SIZE,(BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_SIZE]for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1]for i in ix])
    return x,y

xb,yb = get_batches("TRAIN")

print(xb.shape)
print(yb.shape)

print(xb)
print(yb)