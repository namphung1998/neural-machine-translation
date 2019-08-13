import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .embeddings import ModelEmbeddings

class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()
        self.embeddings = ModelEmbeddings(embed_size, vocab)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.dropout_rate = dropout_rate

        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=4, bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.h_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

    def forward(self, source, target):
        """

        """
        pass

    def encode(self, source_padded, source_lengths):
        """
        Apply the encoder to a batch of source sentences to obtain encoder hidden states
        Also get the initial states for decoder

        @param source_padded (Tensor): tensor of shape (src_len, b), where b is the batch
                                        size, and src_len is the maximum sentence length
        @param source_lengths (list(int)) List of actual lengths of each sentence in the batch

        @returns enc_hiddens (Tensor) tensor of hidden units of shape (b, src_len, h*2)
        @returns dec_init_state (tuple(Tensor, Tensor)): decoder hidden state and cell
        """

        X = self.embeddings.src(source_padded)
        X = pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        enc_hiddens = pad_packed_sequence(enc_hiddens, batch_first=True)[0]

        init_decoder_hidden = self.h_projection(torch.cat([last_hidden[0], last_hidden[1]], dim=-1))
        init_decoder_cell = self.c_projection(torch.cat([last_cell[0], last_cell[1]], dim=-1))

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded):
        """
        Find the combined output vector for a given batch

        @param enc_hiddens (Tensor): Tensor of size (b, src_len, h*2)
        @param enc_masks (Tensor): Tensor of size (b, src_len)
        @param dec_init_state (tuple(Tensor, Tensor)) Initial hidden state and cell for decoder (see encoder function)
        @param target_padded (Tensor) Ground-truth target sentence of size (tgt_len, b)

        @returns combined_outputs (Tensor) tensor of size (tgt_len, b, b)
        """
        target_padded = target_padded[:-1]

        dec_state = dec_init_state
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        combined_output = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.embeddings.target(target_padded)

        for y_t in torch.split(Y, 1, dim=0):
            y_t = torch.squeeze(y_t, dim=0)
            y_bar_t = torch.cat([y_t, o_prev], dim=-1)
            dec_init_state, o_t, _ = self.decoder_step(y_bar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_output.append(o_t)
            o_prev = o_t
        combined_output = torch.stack(combined_output)

        return combined_output

    def decoder_step(self, y_bar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
        """
        Compute one forward step of the LSTM decoder

        @param y_bar_t (Tensor) Tensor of size (b, e + h), concatenation of y_t and o_{t-1}
        @param dec_state (tuple(Tensor, Tensor)) Tuple of tensors both of size (b, h) containing
                                                the previous hidden state and cell, respectively
        @param enc_hiddens (Tensor) Tensor of size (b, src_len, h*2)
        @param enc_hiddens_proj (Tensor) Tensor of size (b, src_len, h) Encoder hidden state projected
                                        from h * 2 to h.
        @param enc_masks (Tensor): Tensor of size (b, src_len) masks

        @returns dec_state (tuple(Tensor, Tensor)) tuple of tensors bothh of size (b, h) containing
                                                    the new hidden state and cell, respectively
        @returns combined_output (Tensor) tensor of size (b, h) combined output vector at time t
        @returns e_t (Tensor) Tensor of size (b, src_len) attention scores distribution
        """
        dec_hidden, dec_cell = self.decoder(y_bar_t, dec_state)
        e_t = torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, 2))
        e_t = torch.unsqueeze(e_t, 2)

        if enc_masks:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        alpha_t = F.softmax(e_t, dim=1)
        a_t = torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens)
        a_t = torch.squeeze(a_t, 1)
        U_t = torch.cat([dec_hidden, a_t], dim=-1)
        V_t = self.combined_output_projection(U_t)
        O_t = torch.tanh(V_t)
        O_t = self.dropout(O_t)

        return (dec_hidden, dec_cell), combined_output, e_t