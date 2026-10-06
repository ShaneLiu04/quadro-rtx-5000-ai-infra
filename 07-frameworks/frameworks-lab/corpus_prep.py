"""corpus_prep.py — build a local docs+code corpus and char tokenizer (AR007 T001).

No network: aggregates text from the local upstream clones (bitsandbytes
python+csrc sources, tinygrad sources, nanoGPT sources). Upstream repos are
read-only (zero modification). Outputs: data/corpus.txt, data/tokens.npy
(uint16), data/corpus_stats.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

LAB = Path(__file__).parent
INFRA = LAB.parent.parent
DATA = LAB / "data"

SOURCES = {
    "bnb_python": [INFRA / "07-frameworks" / "bitsandbytes" / "bitsandbytes", ("*.py", "**/*.py")],
    "bnb_docs": [INFRA / "07-frameworks" / "bitsandbytes", ("*.md",)],
    "bnb_csrc": [INFRA / "07-frameworks" / "bitsandbytes" / "csrc", ("*.cu", "*.cpp", "*.h", "**/*.cu", "**/*.cpp", "**/*.h")],
    "tinygrad": [INFRA / "07-frameworks" / "tinygrad" / "tinygrad", ("*.py", "**/*.py")],
    "nanogpt": [INFRA / "06-training" / "nanoGPT", ("*.py", "*.md")],
}


def collect():
    texts, per_group = [], {}
    for group, (root, patterns) in SOURCES.items():
        files = sorted({p for pat in patterns for p in root.glob(pat)})
        chars = 0
        for p in files:
            t = p.read_text(encoding="utf-8", errors="ignore")
            texts.append(t)
            chars += len(t)
        per_group[group] = {"files": len(files), "chars": chars}
        print(f"[{group}] {len(files)} files, {chars/1024:.0f} KB")
    return "\n".join(texts), per_group


def main():
    DATA.mkdir(exist_ok=True)
    text, per_group = collect()
    assert len(text) > 400_000, f"corpus too small: {len(text)}"

    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    n = len(text)

    freq = Counter(text)
    probs = np.array([freq[c] for c in vocab], dtype=np.float64)
    probs /= probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum())

    tokens = np.fromiter((stoi[c] for c in text), dtype=np.uint16, count=n)
    assert tokens.max() < 65536

    # encode/decode roundtrip gate on a slice
    sample = text[10_000:11_000]
    rt = "".join(itos[int(t)] for t in np.fromiter((stoi[c] for c in sample), dtype=np.uint16))
    assert rt == sample, "roundtrip failed"

    n_val = max(1, int(n * 0.05))
    split = n - n_val  # val = last 5%

    np.save(DATA / "tokens.npy", tokens)
    (DATA / "corpus.txt").write_text(text, encoding="utf-8")

    stats = {
        "exp": "corpus",
        "total_chars": n,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "char_entropy_bits": round(entropy, 4),
        "per_group": per_group,
        "train_chars": split,
        "val_chars": n_val,
        "split_index": split,
        "tokens_dtype": "uint16",
    }
    (DATA / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    # gates
    assert 64 <= len(vocab) <= 512, f"unexpected vocab size {len(vocab)}"
    assert split > 0 and n_val > 0
    print(f"[corpus] {n} chars, vocab {len(vocab)}, entropy {entropy:.3f} bits/char")
    print(f"[corpus] train {split} / val {n_val}")
    print("[gate] roundtrip exact PASS; vocab deterministic PASS")
    print(json.dumps(per_group))


if __name__ == "__main__":
    main()
