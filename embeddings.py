import torch.nn as nn

class ModelEmbeddings(nn.Module):
    def __init__(self, embed_size, vocab):
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        src_pad_idx = self.src['<pad>']
        tgt_pad_idx = self.src['<pad>']

        self.src = nn.Embedding(len(vocab.src), embed_size, padding_idx=src_pad_idx)
        self.tgt = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=tgt_pad_idx)

