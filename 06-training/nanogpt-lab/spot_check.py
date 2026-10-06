"""AR006 T013 wrap-up spot check (does NOT overwrite any archived result JSON).

1. Re-run the SDPA backend probe and compare with archived results/e7.json.
2. Fresh 200-step fp16 run of the 10.5M baseline config; tok/s must be within
   +/-15% of e1.json (same config, same seed family, step time is regime-stable).
3. Re-run verify_numbers.py (84 doc-number assertions) via subprocess.

Writes results/spot_check.json with gates; asserts all gates pass.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpt_lab  # noqa: E402  (module load: corpus + upstream nanoGPT model)

RES = os.path.join(HERE, "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


out = {"exp": "t013_spot_check"}

# --- 1. SDPA probe replication (compare with archived e7.json, no overwrite)
from torch.profiler import profile, ProfilerActivity  # noqa: E402

arch = load("e7.json")
probe = {}
for prec in ("fp16", "fp32"):
    torch.manual_seed(0)
    model = gpt_lab.GPT(gpt_lab.GPTConfig(block_size=256, vocab_size=gpt_lab.VOCAB,
                                          dropout=0.0, bias=False, **gpt_lab.CONFIGS["3M"]))
    model.cuda().eval()
    x, _ = gpt_lab.get_batch("val", 8, 256, np.random.default_rng(1))
    amp = torch.float16 if prec == "fp16" else None
    with torch.no_grad():
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=amp, enabled=amp is not None):
                model(x)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=amp, enabled=amp is not None):
                    model(x)
    names = [e.key for e in prof.key_averages() if e.self_device_time_total > 0]
    joined = " | ".join(names)
    probe[prec] = {"flash": "flash" in joined.lower(),
                   "mem_efficient": ("efficient" in joined.lower()) or ("fmha" in joined.lower())}
same_backend = all(
    probe[p]["flash"] == arch[p]["flash"] and
    probe[p]["mem_efficient"] == arch[p]["mem_efficient"] for p in ("fp16", "fp32"))
out["sdpa_probe"] = {"fresh": probe,
                     "archived": {p: {"flash": arch[p]["flash"],
                                      "mem_efficient": arch[p]["mem_efficient"]}
                                  for p in ("fp16", "fp32")},
                     "identical_to_archived": same_backend}

# --- 2. Fresh 200-step fp16 run vs e1.json throughput
e1 = load("e1.json")
_, r = gpt_lab.train_one(gpt_lab.CONFIGS["10.5M"], "fp16", steps=200, seed=0)
ratio = r["tokens_per_s"] / e1["tokens_per_s"]
out["fresh_200step"] = {
    "tokens_per_s": r["tokens_per_s"], "step_time_ms_median": r["step_time_ms_median"],
    "final_train_loss_200": r["final_train_loss"],
    "e1_tokens_per_s": e1["tokens_per_s"], "ratio_vs_e1": ratio,
    "within_15pct": abs(ratio - 1.0) <= 0.15,
    "loss_finite_and_decreased": (np.isfinite(r["final_train_loss"])
                                  and r["final_train_loss"] < r["loss_curve"][0]),
}

# --- 3. verify_numbers.py re-run
p = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "verify_numbers.py")],
                   capture_output=True, text=True)
out["verify_numbers"] = {"exit_code": p.returncode, "tail": p.stdout.strip().splitlines()[-1]}

out["gates"] = {
    "sdpa_backend_identical": bool(same_backend),
    "fresh_within_15pct_of_e1": bool(out["fresh_200step"]["within_15pct"]),
    "loss_sane": bool(out["fresh_200step"]["loss_finite_and_decreased"]),
    "verify_numbers_all_pass": p.returncode == 0,
}
with open(os.path.join(RES, "spot_check.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["gates"], indent=1))
print("ratio vs e1:", round(ratio, 4))
assert all(out["gates"].values()), out["gates"]
