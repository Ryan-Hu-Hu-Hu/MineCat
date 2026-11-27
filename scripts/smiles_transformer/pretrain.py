"""
pretrain.py — pretrain our own SMILES Transformer (TrfmSeq2seq) from scratch on the
prepared ChEMBL corpus, as a denoising autoencoder (reconstruct the SMILES token
sequence; NLL loss, PAD ignored; RDKit-randomization augmentation in the dataset).

Identical architecture to the reference SMILES-Transformer (hidden=256, 4 enc + 4 dec
layers, nhead=4, dim_ff=256); `encode()` yields the 1,024-d embedding used by MineCat.
The released checkpoint (weights/smiles_transformer/) was trained with the defaults below
(10 epochs, batch 256, lr 1e-4, 15% token masking).

Run (from repo root):
  python scripts/smiles_transformer/pretrain.py --corpus smiles_corpus/chembl37_smiles.txt \
      --vocab smiles_corpus/vocab.json --ckpt-dir smiles_corpus/checkpoints

Checkpoints -> <ckpt-dir>/smiles_transformer_best.pt (+ one checkpoint per epoch).
"""
import argparse
import math
import os
import sys
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from minecat.smiles_transformer.tokenizer import WordVocab, PAD      # noqa: E402
from minecat.smiles_transformer.model import TrfmSeq2seq              # noqa: E402
from minecat.smiles_transformer.dataset import Seq2seqDataset         # noqa: E402


def _hparams(args):
    return {k: getattr(args, k) for k in ('epochs', 'batch_size', 'lr', 'hidden', 'n_layer', 'seq_len', 'p_mask')}


def load_smiles(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _ar_batch(batch, device):
    """batch = (src_b, tgt_b), each (B, T) padded ids -> denoising-translation AR tensors.
    src    = randomized+masked source (T, B)   encoder input
    tgt_in = canonical[:-1]  (T-1, B)          decoder input (starts with SOS)
    tgt_out= canonical[1:]   (T-1, B)          targets (next-token)
    + bool causal mask and padding masks."""
    src_b, tgt_b = batch
    src_b = src_b.to(device, non_blocking=True)   # (B, T)
    tgt_b = tgt_b.to(device, non_blocking=True)   # (B, T)
    src = src_b.t().contiguous()                  # (T, B)
    tgt_in = tgt_b[:, :-1].t().contiguous()       # (T-1, B)
    tgt_out = tgt_b[:, 1:].t().contiguous()       # (T-1, B)
    src_kpm = (src_b == PAD)                       # (B, T)
    tgt_kpm = (tgt_b[:, :-1] == PAD)              # (B, T-1)
    Tm = tgt_in.size(0)
    # bool causal mask (True = disallowed) to match the bool padding masks.
    tgt_mask = torch.triu(torch.ones(Tm, Tm, dtype=torch.bool, device=device), diagonal=1)
    return src, tgt_in, tgt_out, src_kpm, tgt_kpm, tgt_mask


@torch.no_grad()
def evaluate(model, loader, vocab_size, device):
    model.eval()
    tot, nb = 0.0, 0
    for sm in loader:
        src, tgt_in, tgt_out, src_kpm, tgt_kpm, tgt_mask = _ar_batch(sm, device)
        out = model(src, tgt_in, tgt_mask=tgt_mask,
                    src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm)  # (T-1,B,V)
        loss = nn.functional.nll_loss(
            out.reshape(-1, vocab_size), tgt_out.reshape(-1), ignore_index=PAD)
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--vocab', required=True)
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--n_layer', type=int, default=4)
    ap.add_argument('--seq_len', type=int, default=220)
    ap.add_argument('--p_mask', type=float, default=0.15,
                    help='fraction of encoder-input tokens masked (denoising)')
    ap.add_argument('--n_worker', type=int, default=8)
    ap.add_argument('--test_size', type=int, default=10000)
    ap.add_argument('--eval_every', type=int, default=2000, help='iters between val/ckpt')
    args = ap.parse_args()

    CKPT_DIR = args.ckpt_dir
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = WordVocab.load(args.vocab)
    V = len(vocab)
    print(f"vocab={V}  device={device}")

    smiles = load_smiles(args.corpus)
    print(f"corpus: {len(smiles):,} SMILES")
    ds = Seq2seqDataset(smiles, vocab, seq_len=args.seq_len, p_mask=args.p_mask)
    test_size = min(args.test_size, len(ds) // 20)
    train, test = random_split(ds, [len(ds) - test_size, test_size],
                               generator=torch.Generator().manual_seed(2025))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_worker, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.n_worker, pin_memory=True)
    print(f"train={len(train):,}  test={len(test):,}")

    model = TrfmSeq2seq(V, args.hidden, V, args.n_layer).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best = None
    for e in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for b, sm in enumerate(train_loader):
            src, tgt_in, tgt_out, src_kpm, tgt_kpm, tgt_mask = _ar_batch(sm, device)
            opt.zero_grad()
            out = model(src, tgt_in, tgt_mask=tgt_mask,
                        src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm)
            loss = nn.functional.nll_loss(
                out.reshape(-1, V), tgt_out.reshape(-1), ignore_index=PAD)
            loss.backward()
            opt.step()
            if b % 200 == 0:
                print(f"E{e} it{b}/{len(train_loader)} loss {loss.item():.3f} "
                      f"ppl {math.exp(min(loss.item(),20)):.2f} "
                      f"({(time.time()-t0)/max(b,1):.3f}s/it)", flush=True)
            if b > 0 and b % args.eval_every == 0:
                vl = evaluate(model, test_loader, V, device)
                print(f"[val] E{e} it{b} loss {vl:.3f} ppl {math.exp(min(vl,20)):.2f}", flush=True)
                if best is None or vl < best:
                    best = vl
                    torch.save({'model': model.state_dict(), 'args': _hparams(args),
                                'vocab_size': V, 'val_loss': vl},
                               os.path.join(CKPT_DIR, 'smiles_transformer_best.pt'))
                    print(f"  [*] saved best (val {vl:.3f})", flush=True)
                model.train()
        vl = evaluate(model, test_loader, V, device)
        torch.save({'model': model.state_dict(), 'args': _hparams(args),
                    'vocab_size': V, 'val_loss': vl},
                   os.path.join(CKPT_DIR, f'smiles_transformer_epoch{e}.pt'))
        print(f"== epoch {e} done  val {vl:.3f}  ({(time.time()-t0)/60:.1f} min) ==", flush=True)
        if best is None or vl < best:
            best = vl
            torch.save({'model': model.state_dict(), 'args': _hparams(args),
                        'vocab_size': V, 'val_loss': vl},
                       os.path.join(CKPT_DIR, 'smiles_transformer_best.pt'))
            print(f"  [*] saved best (val {vl:.3f})", flush=True)
    print("PRETRAIN DONE  best val:", best)


if __name__ == '__main__':
    main()
