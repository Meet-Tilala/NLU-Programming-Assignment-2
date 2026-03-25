import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ── Special token definitions ────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"   # index 0  (conventional choice for nn.Embedding padding_idx)
SOS_TOKEN = "<SOS>"   # index 1  start-of-sequence marker
EOS_TOKEN = "<EOS>"   # index 2  end-of-sequence / stop marker


class CharVocab:
    def __init__(self, names: list[str]):
        # Collect every unique character across all names
        chars = sorted(set("".join(names)))

        # Special tokens occupy the first indices for clarity
        self.tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + chars

        # Forward (char -> idx) and inverse (idx -> char) mappings
        self.char2idx: dict[str, int] = {c: i for i, c in enumerate(self.tokens)}
        self.idx2char: dict[int, str] = {i: c for i, c in enumerate(self.tokens)}

        # Convenience aliases
        self.pad_idx = self.char2idx[PAD_TOKEN]
        self.sos_idx = self.char2idx[SOS_TOKEN]
        self.eos_idx = self.char2idx[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, name: str) -> list[int]:
        """Convert a name string to a list of integer indices (no special tokens)."""
        return [self.char2idx[c] for c in name if c in self.char2idx]

    def decode(self, indices: list[int]) -> str:
        """Convert integer indices back to a name string, skipping special tokens."""
        return "".join(
            self.idx2char[i]
            for i in indices
            if i not in (self.pad_idx, self.sos_idx, self.eos_idx)
        )


class NamesDataset(Dataset):

    def __init__(self, names: list[str], vocab: CharVocab):
        self.vocab = vocab
        self.samples: list[tuple[list[int], list[int]]] = []

        for name in names:
            encoded = vocab.encode(name)
            if len(encoded) == 0:
                continue   # skip empty names (e.g. after stripping unknown chars)

            input_seq  = [vocab.sos_idx] + encoded          # [<SOS>, c1, c2, ..., cn]
            target_seq = encoded          + [vocab.eos_idx]  # [c1, c2, ..., cn, <EOS>]
            self.samples.append((input_seq, target_seq))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp, tgt = self.samples[idx]
        # Return as LongTensors for embedding lookup
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    # pad_sequence expects a list of 1-D tensors
    inputs_padded  = pad_sequence(inputs,  batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded


def load_names(filepath: str) -> list[str]:
    """Read names from a text file (one name per line), stripping whitespace."""
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def get_dataloader(names: list[str], vocab: CharVocab,
                   batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """Convenience wrapper: build Dataset and wrap in a DataLoader."""
    dataset = NamesDataset(names, vocab)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn)
