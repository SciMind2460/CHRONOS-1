import torch.nn as nn

class TextGenerationTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super(TextGenerationTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output
