"""
tokenizer.py — atom-aware SMILES tokenizer + vocabulary for our own SMILES
Transformer (a clean, in-house reimplementation of the published SMILES-Transformer
tokenization scheme, Honda et al. 2019).

The tokenizer, vocabulary and trained weights are all in-house (no third-party weights).

Special-token indices are FIXED and identical to the reference scheme so downstream
id conventions match:
    PAD=0  UNK=1  EOS=2  SOS=3  MASK=4
"""
import json
import pickle
from collections import Counter

PAD, UNK, EOS, SOS, MASK = 0, 1, 2, 3, 4
SPECIALS = ['<pad>', '<unk>', '<eos>', '<sos>', '<mask>']

# Two-character atoms/ions that must not be split into single characters.
# (Superset of the reference list; extra entries are harmless — they only matter
#  when the corresponding two-char sequence actually appears.)
_TWO_CHAR = {
    'Cl', 'Ca', 'Cu', 'Br', 'Be', 'Ba', 'Bi', 'Si', 'Se', 'Sr', 'Na', 'Ni',
    'Rb', 'Ra', 'Xe', 'Li', 'Al', 'As', 'Ag', 'Au', 'Mg', 'Mn', 'Te', 'Zn',
    'si', 'se', 'te', 'He', 'Kr', 'Fe',
    '+2', '+3', '+4', '-2', '-3', '-4',
}
# Leading characters that trigger a two-character lookahead.
_LEAD = set(c[0] for c in _TWO_CHAR) | {'%'}


def split(sm: str) -> str:
    """Split a SMILES string into space-separated tokens, keeping multi-char
    atoms (Cl, Br, Si, ...), ring closures (%NN) and charged ions (+2, -3) intact.
    Returns a single space-joined string (mirrors the reference `split`)."""
    arr = []
    i = 0
    n = len(sm)
    while i < n - 1:
        c = sm[i]
        if c == '%':
            arr.append(sm[i:i + 3]); i += 3
        elif c in _LEAD and sm[i:i + 2] in _TWO_CHAR:
            arr.append(sm[i:i + 2]); i += 2
        else:
            arr.append(c); i += 1
    if i == n - 1:
        arr.append(sm[i])
    return ' '.join(arr)


class WordVocab:
    """Minimal frequency-ranked vocabulary over whitespace-tokenized SMILES.
    itos[0:5] are always the special tokens above."""

    def __init__(self, token_lines, max_size=None, min_freq=1):
        counter = Counter()
        for line in token_lines:
            toks = line if isinstance(line, list) else line.replace('\n', '').split()
            counter.update(toks)
        for t in SPECIALS:
            counter.pop(t, None)
        self.itos = list(SPECIALS)
        words = sorted(counter.items(), key=lambda kv: kv[0])
        words.sort(key=lambda kv: kv[1], reverse=True)
        cap = None if max_size is None else max_size + len(self.itos)
        for w, f in words:
            if f < min_freq or len(self.itos) == cap:
                break
            self.itos.append(w)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_index, self.unk_index = PAD, UNK
        self.eos_index, self.sos_index, self.mask_index = EOS, SOS, MASK

    def __len__(self):
        return len(self.itos)

    def to_ids(self, tokens):
        return [self.stoi.get(t, self.unk_index) for t in tokens]

    def save(self, path):
        """Save as JSON (token list in index order)."""
        with open(path, 'w') as f:
            json.dump({'itos': self.itos}, f, indent=0)

    @classmethod
    def from_itos(cls, itos):
        v = cls([], min_freq=1)
        v.itos = list(itos)
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v

    @classmethod
    def load(cls, path):
        """Load a JSON vocabulary (or a legacy pickled WordVocab)."""
        if path.endswith('.json'):
            with open(path) as f:
                return cls.from_itos(json.load(f)['itos'])
        with open(path, 'rb') as f:
            obj = _LegacyUnpickler(f).load()
        return cls.from_itos(obj.itos)


class _LegacyUnpickler(pickle.Unpickler):
    """Resolve WordVocab pickled under its original top-level module name."""

    def find_class(self, module, name):
        if name == 'WordVocab':
            return WordVocab
        return super().find_class(module, name)
