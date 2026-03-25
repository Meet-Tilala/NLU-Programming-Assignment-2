"""
PA-2 | Problem 2 | TASK-1: Model Implementations
==================================================
Three character-level generative models, all implemented in PyTorch:

  1. VanillaRNN       — single-layer Elman RNN
  2. BidirectionalLSTM (BLSTM) — bidirectional LSTM encoder + unidirectional LSTM decoder
  3. RNNWithAttention — unidirectional LSTM with a Bahdanau-style additive attention mechanism

All models share the same interface:
    forward(x, hidden=None) -> (logits, hidden)
    init_hidden(batch_size)  -> initial hidden state tensor(s)
    generate(vocab, max_len, temperature) -> str
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataset import CharVocab


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Vanilla RNN
# ═══════════════════════════════════════════════════════════════════════════════

class VanillaRNN(nn.Module):
    """
    Single-layer Elman (Vanilla) RNN for character-level sequence generation.

    Architecture:
        Embedding  ->  RNN cell  ->  Linear (projection to vocab)

    At each timestep t:
        h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
        y_t = Linear(h_t)                 [unnormalised logits over vocab]

    The model is purely unidirectional, which is natural for generation:
    we only condition on past characters, never future ones.

    Parameters:
        vocab_size  — number of unique characters (including special tokens)
        embed_dim   — size of the character embedding vector
        hidden_size — number of units in the RNN hidden state
        num_layers  — number of stacked RNN layers
        dropout     — dropout probability (applied between stacked layers)
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Character embedding: maps integer token ids -> dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Elman RNN core; batch_first=True means input shape is (B, T, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout needs >=2 layers
            nonlinearity="tanh"
        )

        # Dropout applied to the RNN output before the final projection
        self.dropout = nn.Dropout(dropout)

        # Final linear layer maps hidden state -> unnormalised logits over all chars
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Forward pass.

        Args:
            x      : (batch, seq_len) integer token ids
            hidden : (num_layers, batch, hidden_size) or None

        Returns:
            logits : (batch, seq_len, vocab_size)
            hidden : updated hidden state (num_layers, batch, hidden_size)
        """
        # Embed tokens -> (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # Run through RNN -> output: (batch, seq_len, hidden_size)
        output, hidden = self.rnn(embedded, hidden)

        # Project to vocabulary size -> (batch, seq_len, vocab_size)
        logits = self.fc(self.dropout(output))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return a zero hidden state for the start of a sequence."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, vocab: CharVocab, max_len: int = 15,
                 temperature: float = 1.0, device: torch.device = torch.device("cpu")) -> str:
        """
        Autoregressively generate a single name.

        At each step:
          1. Feed the previous character (starting with <SOS>)
          2. Get logits from the model
          3. Apply temperature scaling and sample from the softmax distribution
          4. Stop when <EOS> is predicted or max_len is reached

        Temperature controls sharpness:
          T < 1  → more deterministic (picks high-probability chars)
          T > 1  → more random (flatter distribution)
        """
        self.eval()
        hidden = self.init_hidden(1, device)           # batch_size = 1
        current_char = torch.tensor([[vocab.sos_idx]], device=device)  # (1, 1)
        generated = []

        for _ in range(max_len):
            logits, hidden = self.forward(current_char, hidden)   # (1, 1, vocab_size)
            logits = logits[:, -1, :] / temperature               # (1, vocab_size)

            # Sample from the distribution (more interesting than greedy argmax)
            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, 1).item()

            if next_char == vocab.eos_idx:
                break
            generated.append(next_char)
            current_char = torch.tensor([[next_char]], device=device)

        return vocab.decode(generated)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Bidirectional LSTM (BLSTM)
# ═══════════════════════════════════════════════════════════════════════════════

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM encoder + unidirectional LSTM decoder for name generation.

    Motivation:
        During TRAINING we have the full name available, so a BiLSTM encoder can
        process it in both directions and build a richer context representation.
        The encoder's final hidden state (concatenation of forward + backward)
        is projected down and used to initialise the unidirectional decoder.

    During INFERENCE (generate()), the encoder is not available because we don't
    have the full name yet.  We fall back to a zero-initialised decoder hidden
    state and generate autoregressively — identical to VanillaRNN generation.

    Architecture (training):
        [Encoder]
        Embedding -> BiLSTM  ->  (h_fwd, h_bwd) concatenated per layer
                              ->  Linear projection -> decoder initial state

        [Decoder]
        Embedding -> LSTM   ->  Linear  ->  logits

    Parameters:
        vocab_size   — vocabulary size
        embed_dim    — embedding dimension (shared by encoder and decoder)
        hidden_size  — hidden units per LSTM direction in encoder;
                       decoder hidden size = hidden_size (after projection)
        num_layers   — number of stacked LSTM layers
        dropout      — dropout probability
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Shared embedding layer (encoder and decoder use the same character space)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        # ── Encoder ──────────────────────────────────────────────────────────
        # bidirectional=True means the output has 2 * hidden_size features
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Project concatenated bidirectional hidden & cell states to decoder size
        # The encoder outputs (h, c) each of shape (2*num_layers, B, hidden_size).
        # We reshape to (num_layers, B, 2*hidden_size) and project to hidden_size.
        self.hidden_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.cell_proj   = nn.Linear(2 * hidden_size, hidden_size)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection: hidden state -> vocabulary logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, x: torch.Tensor):
        """
        Run the bidirectional encoder over a full input sequence.

        Returns the projected (h, c) pair suitable for initialising the decoder.
        """
        embedded = self.dropout(self.embedding(x))      # (B, T, embed_dim)
        _, (h_n, c_n) = self.encoder(embedded)
        # h_n shape: (2 * num_layers, B, hidden_size) — interleaved fwd/bwd
        # Reshape so we can project each layer independently
        B = x.size(0)
        # Stack forward and backward directions: (num_layers, B, 2*hidden_size)
        h_n = h_n.view(self.num_layers, 2, B, self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, B, self.hidden_size)
        h_cat = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=-1)  # (L, B, 2H)
        c_cat = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=-1)
        # Project to decoder hidden size
        h_dec = torch.tanh(self.hidden_proj(h_cat))   # (L, B, H)
        c_dec = torch.tanh(self.cell_proj(c_cat))
        return h_dec, c_dec

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Forward pass: encode input sequence, then decode it (teacher forcing).

        In training, x contains the full [<SOS>, c1, ..., cn] sequence.
        The encoder sees the full sequence; the decoder is initialised from
        the encoder state.

        Args:
            x      : (batch, seq_len) — input token ids
            hidden : unused (state is always derived from encoder during training)

        Returns:
            logits : (batch, seq_len, vocab_size)
            hidden : final decoder (h, c) state
        """
        # Encode -> get initial decoder state
        h_dec, c_dec = self.encode(x)

        # Decode using teacher forcing (feed the same x to the decoder)
        embedded = self.dropout(self.embedding(x))         # (B, T, embed_dim)
        output, (h_dec, c_dec) = self.decoder(embedded, (h_dec, c_dec))
        logits = self.fc(self.dropout(output))             # (B, T, vocab_size)
        return logits, (h_dec, c_dec)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Zero-initialised decoder state (used during generation only)."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, vocab: CharVocab, max_len: int = 15,
                 temperature: float = 1.0, device: torch.device = torch.device("cpu")) -> str:
        """Autoregressive generation using the decoder only (no encoder input available)."""
        self.eval()
        hidden = self.init_hidden(1, device)
        current_char = torch.tensor([[vocab.sos_idx]], device=device)
        generated = []

        for _ in range(max_len):
            embedded = self.dropout(self.embedding(current_char))   # (1, 1, embed_dim)
            output, hidden = self.decoder(embedded, hidden)         # (1, 1, hidden_size)
            logits = self.fc(output[:, -1, :]) / temperature        # (1, vocab_size)

            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, 1).item()

            if next_char == vocab.eos_idx:
                break
            generated.append(next_char)
            current_char = torch.tensor([[next_char]], device=device)

        return vocab.decode(generated)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — RNN with Bahdanau Attention
# ═══════════════════════════════════════════════════════════════════════════════

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.

    Reference: Bahdanau et al. (2015) "Neural Machine Translation by
    Jointly Learning to Align and Translate".

    Given:
        query   : current decoder hidden state  (B, hidden_size)
        keys    : all encoder hidden states     (B, T, hidden_size)

    The attention score for timestep t is:
        score(t) = v^T * tanh(W_q * query + W_k * key_t)

    Attention weights are the softmax over all scores.
    The context vector is the weighted sum of the keys.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Learned linear transformations for query and keys
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key   = nn.Linear(hidden_size, hidden_size, bias=False)
        # Energy scalar: projects tanh output to a single score per timestep
        self.v       = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Compute context vector and attention weights.

        Args:
            query : (B, H)    — current decoder state
            keys  : (B, T, H) — encoder outputs
            mask  : (B, T) bool tensor; True where keys should be masked (PAD positions)

        Returns:
            context : (B, H) weighted sum of encoder outputs
            weights : (B, T) attention distribution (sums to 1)
        """
        # Expand query to match keys' time dimension: (B, 1, H)
        query_expanded = self.W_query(query).unsqueeze(1)  # (B, 1, H)
        keys_proj = self.W_key(keys)                        # (B, T, H)

        # Additive scoring: (B, T, H) -> (B, T, 1) -> (B, T)
        energy = self.v(torch.tanh(query_expanded + keys_proj)).squeeze(-1)

        # Mask PAD positions with -inf so they receive zero attention weight
        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        weights = F.softmax(energy, dim=-1)                 # (B, T)

        # Weighted sum of encoder outputs: (B, T) @ (B, T, H) -> (B, H)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class RNNWithAttention(nn.Module):
    """
    Unidirectional LSTM with Bahdanau attention for character-level generation.

    Architecture:
        [Encoder]  Embedding -> LSTM  ->  encoder_outputs (B, T, H)

        [Decoder]  At each step t:
            1. Embed current character: x_t (B, embed_dim)
            2. Compute attention context over encoder_outputs using h_{t-1}
            3. Concatenate [x_t, context] and feed to decoder LSTM cell
            4. Project hidden state to logits

    The attention mechanism allows the decoder to dynamically focus on
    different parts of the input sequence at each generation step.

    Parameters:
        vocab_size   — vocabulary size
        embed_dim    — embedding dimension
        hidden_size  — encoder and decoder hidden state size
        num_layers   — number of stacked LSTM layers (encoder and decoder)
        dropout      — dropout probability
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        # ── Encoder (unidirectional) ──────────────────────────────────────────
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = BahdanauAttention(hidden_size)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Input = embedding + context vector concatenated => embed_dim + hidden_size
        self.decoder = nn.LSTM(
            input_size=embed_dim + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Final projection to vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, x: torch.Tensor):
        """
        Encode the full input sequence.

        Returns:
            encoder_outputs : (B, T, H) — hidden state at every timestep
            (h_n, c_n)      : final hidden and cell states
        """
        embedded = self.dropout(self.embedding(x))
        encoder_outputs, (h_n, c_n) = self.encoder(embedded)
        return encoder_outputs, (h_n, c_n)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Forward pass with attention (teacher forcing).

        At each decoder timestep t:
          - attend over all encoder outputs using the previous decoder hidden state
          - concatenate the current embedding with the context vector
          - run one step of the decoder LSTM

        Args:
            x      : (B, T) input token ids  [<SOS>, c1, ..., cn]
            hidden : unused; state is initialised from the encoder

        Returns:
            logits : (B, T, vocab_size)
            hidden : final (h, c) decoder state
        """
        B, T = x.shape

        # ── Encode the full input ─────────────────────────────────────────────
        encoder_outputs, (h_n, c_n) = self.encode(x)

        # Build a PAD mask: True where x == PAD_TOKEN (idx 0)
        pad_mask = (x == 0)                                    # (B, T)

        # ── Decode step-by-step ───────────────────────────────────────────────
        embedded   = self.dropout(self.embedding(x))           # (B, T, embed_dim)
        h_dec, c_dec = h_n, c_n                               # init decoder from encoder
        all_logits = []

        for t in range(T):
            # Current decoder query: top layer of h_dec, shape (B, H)
            query = h_dec[-1]                                  # (B, H)

            # Compute attention context using encoder outputs
            context, _ = self.attention(query, encoder_outputs, mask=pad_mask)
            # context: (B, H)

            # Concatenate character embedding with context: (B, 1, embed_dim + H)
            dec_input = torch.cat([embedded[:, t:t+1, :],
                                   context.unsqueeze(1)], dim=-1)

            # One decoder LSTM step
            dec_output, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
            # dec_output: (B, 1, H)

            logits_t = self.fc(self.dropout(dec_output))       # (B, 1, vocab_size)
            all_logits.append(logits_t)

        logits = torch.cat(all_logits, dim=1)                  # (B, T, vocab_size)
        return logits, (h_dec, c_dec)

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, vocab: CharVocab, max_len: int = 15,
                 temperature: float = 1.0, device: torch.device = torch.device("cpu")) -> str:
        """
        Autoregressive generation with a single-token attention context.

        At each step we maintain a short running context (the tokens generated so
        far) so the attention mechanism has something to attend over.
        """
        self.eval()
        generated_ids = [vocab.sos_idx]
        h_dec = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        c_dec = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

        for _ in range(max_len):
            # Build current context sequence from what we've generated so far
            ctx_tensor = torch.tensor([generated_ids], device=device)  # (1, t)

            # Encode the growing context
            encoder_outputs, _ = self.encode(ctx_tensor)               # (1, t, H)

            # Attend using the last decoder hidden state
            query = h_dec[-1]                                           # (1, H)
            context, _ = self.attention(query, encoder_outputs)         # (1, H)

            # Embed only the last generated character
            last_char = torch.tensor([[generated_ids[-1]]], device=device)  # (1, 1)
            emb = self.dropout(self.embedding(last_char))                   # (1, 1, E)

            # Decode one step
            dec_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)     # (1,1,E+H)
            dec_output, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))

            logits = self.fc(dec_output[:, -1, :]) / temperature            # (1, V)
            probs  = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, 1).item()

            if next_char == vocab.eos_idx:
                break
            generated_ids.append(next_char)

        return vocab.decode(generated_ids[1:])   # strip the leading <SOS>
