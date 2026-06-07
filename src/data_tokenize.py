import torch

class DataTokenizer():
    def __init__(self,data_path):
        with open(data_path) as file:
            self.text = file.read()
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for i,ch in enumerate(self.vocab)}
    
    def tokenize(self):
        tokens = []
        for c in self.text:
            tokens.append(self.stoi[c])
        return torch.tensor(tokens)
    
    def decode(self,tokens):
        return "".join(self.itos[int(token)] for token in tokens)
