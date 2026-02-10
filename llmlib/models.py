import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from .config import cfg
from .layers import Head, MultiHeadAttention, FeedForward, Block, BlockNew


class LSTMmodel(nn.Module):
    def __init__(self, n_features, n_target_length, n_layers=3, bidirectional=True, hybrid=True, n_hidden_size=120, dropout=0.3):
        super(LSTMmodel, self).__init__()
        # lstm input shape:
        # unbatched:        (L,Hin)
        # batched:          (L,N,Hin)
        # batch_first=True  (N,L,Hin)
        # L         =       sequence length
        # N         =       batch size
        # Hin       =       input size
        # Hout      =       output size == hidden_size if no projection.
        self.hybrid = hybrid
        if bidirectional:
            self.bid = True
            self.d = 2
            self.h = 2
        else:
            self.bid = False
            self.d = 1
            self.h = 1
        self.n_target_length = n_target_length
        self.drop_out = dropout
        self.hidden_size = n_hidden_size
        self.num_layers = n_layers
        self.n_features = n_features
        if self.hybrid:
            self.h = 1 # replacement for d in dec_lin
            self.cell_align = self.cell_condense_model()
            self.hidden_align = self.hidden_condense_model()

        self.enc = self.encoder()
        self.dec = self.decoder()
        self.dec_lin = self.dec_lin_model()
        self.enc_lin = self.enc_lin_model()


    def cell_condense_model(self):
        cell_align = nn.Sequential()
        cell_align.append(nn.Linear(self.hidden_size * self.d * self.num_layers,
                                      int(self.hidden_size * self.d * self.num_layers * 2)))
        cell_align.append(nn.Dropout(self.drop_out, inplace=False))
        cell_align.append(nn.Tanh())
        cell_align.append(nn.Linear(int(self.hidden_size * self.d * self.num_layers * 2),
                                      int(self.hidden_size * self.d * self.num_layers * 1)))
        cell_align.append(nn.Dropout(self.drop_out, inplace=False))
        cell_align.append(nn.Tanh())
        cell_align.append(nn.Linear(int(self.hidden_size * self.d * self.num_layers * 1),
                                      int(self.hidden_size * self.d * self.num_layers * 0.7)))
        cell_align.append(nn.Dropout(self.drop_out, inplace=False))
        cell_align.append(nn.Tanh())
        cell_align.append(nn.Linear(int(self.hidden_size * self.d * self.num_layers * 0.7),
                                      int(self.hidden_size * self.num_layers)))
        return cell_align

    def hidden_condense_model(self):
        hidden_align = nn.Sequential()
        hidden_align.append(nn.Linear(self.hidden_size * self.d * self.num_layers, int(self.hidden_size * self.d * self.num_layers * 2)))
        hidden_align.append(nn.Dropout(self.drop_out, inplace=False))
        hidden_align.append(nn.Tanh())
        hidden_align.append(nn.Linear(int(self.hidden_size * self.d * self.num_layers * 2), int(self.hidden_size * self.d * self.num_layers * 1)))
        hidden_align.append(nn.Dropout(self.drop_out, inplace=False))
        hidden_align.append(nn.Tanh())
        hidden_align.append(nn.Linear(int(self.hidden_size * self.d * self.num_layers * 1), int(self.hidden_size * self.d * self.num_layers * 0.7)))
        hidden_align.append(nn.Dropout(self.drop_out, inplace=False))
        hidden_align.append(nn.Tanh())
        hidden_align.append(nn.Linear(int(self.hidden_size * self.d * self.num_layers * 0.7),
                                      int(self.hidden_size * self.num_layers)))
        return hidden_align

    def encoder(self):
        enc = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, bidirectional=self.bid, num_layers=self.num_layers,
                        batch_first=True, dropout=self.drop_out*0.5)
        return enc

    def enc_lin_model(self):
        """
        enc_lin = nn.Sequential()
        enc_lin.append(nn.Linear(self.hidden_size*self.d, self.n_features))
        enc_lin.append(nn.Dropout(self.drop_out, inplace=True))"""

        """
        enc_lin.append(nn.Linear(self.hidden_size * self.d, int((self.hidden_size * self.d) / 2)))
        enc_lin.append(nn.Dropout(self.drop_out, inplace=True))
        enc_lin.append(nn.Tanh())
        enc_lin.append(nn.Linear(int((self.hidden_size * self.d) / 2), self.n_features))
        """
        enc_lin = nn.Sequential()
        enc_lin.append(nn.Linear(self.hidden_size * self.d, int((self.hidden_size * self.d) * 2)))
        enc_lin.append(nn.Dropout(self.drop_out, inplace=False))
        enc_lin.append(nn.LayerNorm(int((self.hidden_size * self.d) * 2), eps=1e-5))
        enc_lin.append(nn.Tanh())
        enc_lin.append(nn.Linear(int((self.hidden_size * self.d) * 2), self.hidden_size * self.d))
        enc_lin.append(nn.Dropout(self.drop_out, inplace=False))
        enc_lin.append(nn.LayerNorm(self.hidden_size * self.d, eps=1e-5))
        enc_lin.append(nn.Tanh())
        enc_lin.append(nn.Linear(self.hidden_size * self.d, int((self.hidden_size * self.d) / 2)))
        enc_lin.append(nn.Dropout(self.drop_out, inplace=False))
        enc_lin.append(nn.LayerNorm(int((self.hidden_size * self.d) / 2), eps=1e-5))
        enc_lin.append(nn.Tanh())
        enc_lin.append(nn.Linear(int((self.hidden_size * self.d) / 2), self.n_features))

        return enc_lin

    def decoder(self):
        if self.hybrid:
            dec = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, bidirectional=False,
                           num_layers=self.num_layers, batch_first=True, dropout=self.drop_out*0.2)
        else:
            dec = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, bidirectional=self.bid,
                          num_layers=self.num_layers, batch_first=True, dropout=self.drop_out * 0.2)
        return dec

    def dec_lin_model(self):
        """
        dec_lin = nn.Sequential()
        dec_lin.append(nn.Linear(self.hidden_size*self.d, self.n_features))
        dec_lin.append(nn.Dropout(self.drop_out, inplace=True))"""

        """
        dec_lin.append(nn.Linear(self.hidden_size * self.d, int((self.hidden_size * self.d) / 2)))
        dec_lin.append(nn.Dropout(self.drop_out, inplace=True))
        dec_lin.append(nn.Tanh())
        dec_lin.append(nn.Linear(int((self.hidden_size * self.d)/2), self.n_features))
        """

        dec_lin = nn.Sequential()
        dec_lin.append(nn.Linear(self.hidden_size * self.h, int((self.hidden_size * self.h)*2)))
        dec_lin.append(nn.Dropout(self.drop_out, inplace=False))
        dec_lin.append(nn.LayerNorm(int((self.hidden_size * self.h)*2), eps=1e-5))
        dec_lin.append(nn.Tanh())
        dec_lin.append(nn.Linear(int((self.hidden_size * self.h)*2), self.hidden_size * self.h))
        dec_lin.append(nn.Dropout(self.drop_out, inplace=False))
        dec_lin.append(nn.LayerNorm(self.hidden_size * self.h, eps=1e-5))
        dec_lin.append(nn.Tanh())
        dec_lin.append(nn.Linear(self.hidden_size * self.h, int((self.hidden_size * self.h) / 2)))
        dec_lin.append(nn.Dropout(self.drop_out, inplace=False))
        dec_lin.append(nn.LayerNorm(int((self.hidden_size * self.h) / 2), eps=1e-5))
        dec_lin.append(nn.Tanh())
        dec_lin.append(nn.Linear(int((self.hidden_size * self.h)/2), self.n_features))
        return dec_lin

    def enc_forward(self, x):
        h0 = torch.zeros(self.num_layers * self.d, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers * self.d, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        enc_out, hidden = self.enc(x, (h0, c0))
        # repeat_vec = lstm_out[:, -1, :].repeat(TARGET_LENGTH, 1)
        out = self.enc_lin(enc_out[:, -1, :])  # Take the last time step's output (lstm_out is (N,L,d*Hout)-shaped)
        out = torch.unsqueeze(out, dim=1)
        return out, hidden

    def dec_forward(self, x, hidden, y=None, teach_force_p=None):
        if teach_force_p is not None:
            tf_p = teach_force_p
        else:
            tf_p = 0.5
        multi_step_out = []
        for i in range(self.n_target_length):
            if i == 0 or y is None:
                dec_out, hidden = self.dec(x, hidden)
            else:
                rand = random.random()
                if rand < tf_p:
                    y_step  = y[:, i - 1, :].unsqueeze(1) # (N, L, n_features), unsqueeze to return (N,1,n_f) instead (N, n_f)
                    dec_out, hidden = self.dec(y_step, hidden)
                else:
                    dec_out, hidden = self.dec(x, hidden)

            x = self.dec_lin(dec_out)
            multi_step_out.append(x)
        out = torch.cat(multi_step_out, dim=1)
        return torch.clamp(out, min=-5.0, max=5.0) #ToDo: Is this a good idea???

    def condense_directions(self, enc_hidden):
        enc_h = enc_hidden[0]  # [2*num_layers, N, H]
        enc_c = enc_hidden[1]  # [2*num_layers, N, H]
        batch_size = enc_h.size(1) # N
        # swap N and 2*num_layers with transpose, then reshape
        enc_h_flat = enc_h.transpose(0, 1).contiguous().view(batch_size, -1)  # [N, 2*num_layers*H]
        enc_c_flat = enc_c.transpose(0, 1).contiguous().view(batch_size, -1)  # same
        dec_h_0 = self.hidden_align(enc_h_flat)   # [N, num_layers * H]
        dec_c_0 = self.cell_align(enc_c_flat)
        # Reshape to: [N, num_layers, H] with view, then transpose to reach [num_layersN, N, H]
        # for decoder input-states.
        dec_h_0 = dec_h_0.view(batch_size, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()
        dec_c_0 = dec_c_0.view(batch_size, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()
        dec_hidden = (dec_h_0, dec_c_0)
        return dec_hidden

    def forward(self, x, y=None, teach_force_p=None):
        enc_output, hidden = self.enc_forward(x)
        if self.hybrid and self.bid:
            hidden = self.condense_directions(hidden)
        out = self.dec_forward(enc_output, hidden, y, teach_force_p=teach_force_p)
        # out = self.dec_forward(torch.unsqueeze(x[:, -1, :], dim=1), hidden)
        return out


# noinspection DuplicatedCode
class BigramBaseLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):
        #idx and targets are both (B, T, C)
        logits = self.token_embedding_table(idx) # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of inices in the current context
        # this function feeds forward all of idx, but the bigram only looks at the last token.
        # This is done to keep it general for the future attention based model.
        for _ in range(max_new_tokens):
            # get the predictions:
            logits , loss = self.forward(idx)
            # focus only on the last timestep:
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx


# noinspection DuplicatedCode
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(cfg.VOCAB_SIZE, cfg.N_EMBD)
        self.position_embedding_table = nn.Embedding(cfg.CONTEXT_LEN, cfg.N_EMBD)
        self.lm_head = nn.Linear(cfg.N_EMBD, cfg.VOCAB_SIZE)

    def forward(self, idx, targets=None):
        #idx and targets are both (B, T, C)
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx) # (B, T, C-embd)
        position_embed = self.position_embedding_table(torch.arange(T, device=next(self.parameters()).device)) # (T, C)
        x = tok_embed + position_embed
        logits = self.lm_head(x) # (B, T, C-voc-size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of inices in the current context
        # this function feeds forward all of idx, but the bigram only looks at the last token.
        # This is done to keep it general for the future attention based model.
        for _ in range(max_new_tokens):
            # get the predictions:
            logits , loss = self.forward(idx)
            # focus only on the last timestep:
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

# noinspection DuplicatedCode
class DecoderAttentionLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.N_EMBD)
        self.position_embedding_table = nn.Embedding(cfg.CONTEXT_LEN, cfg.N_EMBD)
        # self.sa_head = Head(cfg.N_EMBD)
        # self.sa_heads = MultiHeadAttention(4, cfg.N_EMBD//4) # 4 head concat to 4 * 1/4 emdb = embd
        # self.blocks = nn.Sequential(*[Block(cfg.N_EMBD, n_heads=cfg.N_HEADS) for _ in range(cfg.N_LAYERS)])
        self.blocks = nn.Sequential(*[BlockNew(cfg.N_EMBD, n_heads=cfg.N_HEADS) for _ in range(cfg.N_LAYERS)])
        self.lnf = nn.LayerNorm(cfg.N_EMBD)
        # self.ffwd = FeedForward(cfg.N_EMBD) # feed forward -> 'think' on att per node
        self.lm_head = nn.Linear(cfg.N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        #idx and targets are both (B, T, C)
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx) # (B, T, C-embd)
        position_embed = self.position_embedding_table(torch.arange(T, device=next(self.parameters()).device)) # (T, C)
        x = tok_embed + position_embed
        x = self.blocks(x)
        x = self.lnf(x)
        # x = self.ffwd(x)
        logits = self.lm_head(x) # (B, T, C-voc-size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of inices in the current context
        # this function feeds forward all of idx, but the bigram only looks at the last token.
        # This is done to keep it general for the future attention based model.
        for _ in range(max_new_tokens):
            # crop context to the last Block_size tokens
            idx_cond = idx[:, -cfg.CONTEXT_LEN:]
            # get the predictions:
            logits , loss = self.forward(idx_cond)
            # focus only on the last timestep:
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx