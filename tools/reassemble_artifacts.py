#!/usr/bin/env python3
"""重组分片产物（大文件以 .partNNNN 分片入库，本脚本一键还原并校验）。

用法:  python tools/reassemble_artifacts.py
"""
import hashlib
import os
import sys

MANIFEST = {
    "06-training/nanogpt-lab/results/e1_checkpoint.pt":
        "ab684d83a6de1e240633ec52c5f721816e9942b32b3757598dd6477ad040da35",
    "07-frameworks/frameworks-lab/data/corpus.txt":
        "46b11b96be56531a850f224fc4da2878ae222391012aa2c1c067a40035e76e58",
    "07-frameworks/frameworks-lab/data/tokens.npy":
        "a34ad39649d5ed5bba732a619c72b60ccf37a2a2d404b6d03ca0df7d5e6e53dd",
}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok = True
    for rel, want in MANIFEST.items():
        path = os.path.join(root, *rel.split("/"))
        folder, name = os.path.split(path)
        parts = sorted(p for p in os.listdir(folder)
                       if p.startswith(name + ".part"))
        if not parts:
            print("[skip] %s: 未找到分片" % rel)
            continue
        h = hashlib.sha256()
        with open(path, "wb") as out:
            for p in parts:
                data = open(os.path.join(folder, p), "rb").read()
                h.update(data)
                out.write(data)
        good = h.hexdigest() == want
        ok = ok and good
        print("[%s] %s: %d 个分片 -> %.1f MB"
              % ("OK" if good else "SHA256 不匹配", rel, len(parts),
                 os.path.getsize(path) / 1e6))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
