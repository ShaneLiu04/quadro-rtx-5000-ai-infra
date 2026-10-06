"""AR007: assert that every number quoted in results.md / notes / README matches results/*.json.

Writes results/doc_check.json (all_pass flag + per-check detail).
Mirrors AR005/AR006 lesson: document numbers must have code-level evidence.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

checks = []


def load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


e0 = load(RES, "e0_env.json")
e1 = load(RES, "e1_nf4.json")
e2 = load(RES, "e2_llmint8.json")
e3 = load(RES, "e3_adam8bit.json")
e5 = load(RES, "e5_ledger.json")
corpus = load(HERE, "data", "corpus_stats.json")


def check(name, doc, actual, tol=0.005):
    ok = (doc is not None and actual is not None
          and abs(doc - actual) <= tol * max(1.0, abs(actual)))
    checks.append({"check": name, "doc_value": doc, "json_value": actual, "pass": bool(ok)})
    return ok


def check_eq(name, doc, actual):
    ok = doc == actual
    checks.append({"check": name, "doc_value": doc, "json_value": actual, "pass": bool(ok)})
    return ok


# ---- corpus (README: 17.7MB / vocab 142 / entropy 5.732) --------------------
check("corpus total chars (17.66M)", 17664897, corpus["total_chars"], tol=0)
check("corpus vocab (142)", 142, corpus["vocab_size"], tol=0)
check("corpus entropy (5.732)", 5.732, corpus["char_entropy_bits"], tol=0.0005)

# ---- E0 (results.md: 56 nodes / 236 nodes / THREEFRY 4) ---------------------
check("e0 det graph nodes (56)", 56, e0["tinygrad"]["det_graph_nodes"], tol=0)
check("e0 rand graph nodes (236)", 236, e0["tinygrad"]["rand_graph_nodes"], tol=0)
check("e0 THREEFRY count (4)", 4, e0["tinygrad"]["rand_graph_op_hist"]["THREEFRY"], tol=0)
check_eq("e0 bnb degraded (True)", True, e0["bnb"]["degraded"])
check_eq("e0 int_mm cuda bitexact (True)", True, e0["int_mm"]["cuda_bitexact"])
check_eq("e0 nf4 map == our official table (True)", True, e0["bnb"]["nf4_map_consistent"])
check_eq("e0 nf4 map != raw ndtri (False)", False, e0["bnb"]["nf4_map_equals_raw_ndtri"])
check_eq("e0 bnb version (0.50.3.dev0)", "0.50.3.dev0", e0["bnb"]["version"])

# ---- E1 (results.md/notes: 0.092/0.100/0.127/0.377; sweep; real weights) ----
d = e1["gate_detail"]
check("e1 NF4 @64 (0.092)", 0.092, d["nf4_64"], tol=0.005)
check("e1 INT4 @64 (0.100)", 0.100, d["int4_64"], tol=0.005)
check("e1 nf4_raw @64 (0.127)", 0.127, d["nf4_raw_64"], tol=0.005)
check("e1 FP4 @64 (0.377)", 0.377, d["fp4_64"], tol=0.005)
check("e1 NF4 block32 (0.0873)", 0.0873, e1["synth_sweep"]["nf4"]["32"], tol=0.005)
check("e1 NF4 block1024 (0.1043)", 0.1043, e1["synth_sweep"]["nf4"]["1024"], tol=0.005)
check("e1 head kurtosis (8.4)", 8.4, e1["weights"]["head.weight"]["kurtosis"], tol=0.005)
check("e1 head NF4 (0.1046)", 0.1046, e1["weights"]["head.weight"]["rel_rmse_nf4"], tol=0.005)
check("e1 head INT4 (0.1291)", 0.1291, e1["weights"]["head.weight"]["rel_rmse_int4"], tol=0.005)
check_eq("e1 G1 gate (True)", True, e1["gates"]["g1_nf4_beats_int4_on_N01"])
check("e1 absmax mean (3.80)", 3.80, e1["synth_absmax_mean"], tol=0.005)
check_eq("e1 NF4 official endpoints (-1, 1)",
         [-1.0, 1.0], [e1["levels"]["nf4"][0], e1["levels"]["nf4"][-1]])
check_eq("e1 nf4_raw endpoints (~±1.863)",
         [-1.8627, 1.8627], [round(e1["levels"]["nf4_raw"][0], 4),
                             round(e1["levels"]["nf4_raw"][-1], 4)])

# ---- E2 (results.md: sweep / scaling / throughput / layout) -----------------
ts = e2["threshold_sweep"]
check("e2 tau0 err (4.86e-2)", 0.0486, ts["0.0"]["rel_err"], tol=0.005)
check("e2 tau8 err (1.14e-2)", 0.0114, ts["8.0"]["rel_err"], tol=0.005)
check("e2 tau2 err = fp16 (3.6e-4)", 0.00036, ts["2.0"]["rel_err"], tol=0.02)
check("e2 tau16 err (2.19e-2)", 0.0219, ts["16.0"]["rel_err"], tol=0.005)
check("e2 tau32 err (3.96e-2)", 0.0396, ts["32.0"]["rel_err"], tol=0.005)
check_eq("e2 tau2 degenerate frac (1.0)", 1.0, ts["2.0"]["outlier_frac"])
check_eq("e2 tau8 outlier cols (8)", 8, ts["8.0"]["outlier_cols"])
check("e2 tau8 frac (0.0039)", 0.0039, ts["8.0"]["outlier_frac"], tol=0.01)
check("e2 improvement tau0->tau8 (4.2x)", 4.2,
      ts["0.0"]["rel_err"] / ts["8.0"]["rel_err"], tol=0.02)
sc = e2["scaling_compare"]
check("e2 per-tensor (1.76e-2)", 0.0176, sc["per_tensor"], tol=0.005)
check("e2 per-row (1.17e-2)", 0.0117, sc["per_row_act"], tol=0.005)
check("e2 per-row heavytail (4.86e-2)", 0.0486, sc["per_row_heavytail"], tol=0.005)
tp = e2["throughput"]["2048"]
check("e2 fp16 TFLOPS (52.7)", 52.7, tp["fp16_tflops"], tol=0.005)
check("e2 int8 default TOPS (31.1)", 31.1, tp["int8_top_int32"], tol=0.005)
check("e2 int8 e2e TOPS (8.1)", 8.1, tp["int8_e2e_top"], tol=0.01)
check("e2 default ratio (0.59)", 0.59, tp["ratio_kernel"], tol=0.01)
check("e2 e2e ratio (0.15)", 0.15, tp["ratio_e2e"], tol=0.01)
lay = e2["layout_attribution"]
check("e2 TN TOPS (67.9)", 67.9, lay["rowA_colB"]["tops"], tol=0.005)
check("e2 TN vs default (1.83x, layout-paired)", 1.83,
      lay["rowA_colB"]["tops"] / lay["rowA_rowB"]["tops"], tol=0.01)
check("e2 TN vs fp16 (1.29x)", 1.29,
      lay["rowA_colB"]["tops"] / tp["fp16_tflops"], tol=0.01)
check("e2 rowA_rowB TOPS (37.1)", 37.1, lay["rowA_rowB"]["tops"], tol=0.005)
check("e2 colA_rowB TOPS (34.2)", 34.2, lay["colA_rowB"]["tops"], tol=0.005)
check("e2 colA_colB TOPS (36.0)", 36.0, lay["colA_colB"]["tops"], tol=0.005)
check("e2 fp16 baseline err (3.6e-4)", 0.00036, e2["uniform"]["rel_err_fp16"], tol=0.02)
check_eq("e2 G2 gate (False - honest FAIL)", False, e2["gates"]["g2_int8_kernel_ge_1p2x_fp16"])

# ---- E3 (results.md/notes: diff in training noise / 1.1353 vs 1.1561 / 26.09->6.62MB)
check("e3 final loss diff (0.0207)", 0.0207, e3["final_loss_diff"], tol=0.01)
check("e3 fp32 final (1.1353)", 1.1353, e3["fp32"]["final_loss"], tol=0.001)
check("e3 int8 final (1.1561)", 1.1561, e3["int8"]["final_loss"], tol=0.001)
check("e3 fp32 state MB (26.09)", 26.09, e3["fp32"]["optim_state_bytes"] / 1e6, tol=0.005)
check("e3 int8 state MB (6.62)", 6.62, e3["int8"]["optim_state_bytes"] / 1e6, tol=0.005)
check("e3 mem ratio (0.254)", 0.254, e3["optim_mem_ratio_int8_vs_fp32"], tol=0.005)
check("e3 fp32 step ms (33.9)", 33.9, e3["fp32"]["step_ms_median"], tol=0.01)
check("e3 int8 step ms (62.5)", 62.5, e3["int8"]["step_ms_median"], tol=0.01)
check("e3 step time ratio (1.84)", 1.84,
      e3["int8"]["step_ms_median"] / e3["fp32"]["step_ms_median"], tol=0.01)
check_eq("e3 params (3261440)", 3261440, e3["fp32"]["n_params"])
check_eq("e3 G3 gate (True)", True, e3["gates"]["g3_loss_diff_lt_0.05"])

# ---- E5 (results.md/README: 16.78/14.58/8.03/0.5315; 7.5B projection) -------
pp, th, cr = e5["bytes_per_param"], e5["theory_bytes_per_param"], e5["consistency_ratio"]
check("e5 fp32 B/param (16.777)", 16.777, pp["fp32_training"], tol=0.001)
check("e5 amp B/param (14.583)", 14.583, pp["amp_mixed"], tol=0.001)
check("e5 int8 optim B/param (8.032)", 8.032, pp["int8_optim_full_P32_G16"], tol=0.001)
check("e5 nf4 B/param (0.5315)", 0.5315, pp["nf4_params_storage"], tol=0.001)
check("e5 fp32 theory (16.0)", 16.0, th["fp32_training"], tol=0)
check("e5 nf4 theory (0.53125)", 0.53125, th["nf4_params_storage"], tol=0)
check("e5 fp32 ratio (1.049)", 1.049, cr["fp32_training"], tol=0.005)
check("e5 amp ratio (1.042)", 1.042, cr["amp_mixed"], tol=0.005)
pj = e5["projection_7p5B"]
check("e5 7.5B fp32 GB (125.8)", 125.8, pj["fp32_training_GB"], tol=0.005)
check("e5 7.5B amp GB (109.4)", 109.4, pj["amp_mixed_GB"], tol=0.005)
check("e5 7.5B int8 GB (60.2)", 60.2, pj["int8_optim_GB"], tol=0.005)
check("e5 7.5B QLoRA base GB (4.0)", 4.0, pj["qlora_base_nf4_GB"], tol=0.01)
check("e5 7.5B QLoRA total GB (4.4)", 4.4, pj["qlora_total_example_GB"], tol=0.01)
check_eq("e5 G5 gate (True)", True, e5["gates"]["g5_consistency_within_10pct"])
# projection consistency: bytes/param * 7.5e9 must equal projection GB
check("e5 projection = bpp * 7.5e9 (fp32)",
      pj["fp32_training_GB"], pp["fp32_training"] * 7.5e9 / 1e9, tol=0.001)
check("e5 projection = bpp * 7.5e9 (nf4)",
      pj["qlora_base_nf4_GB"], pp["nf4_params_storage"] * 7.5e9 / 1e9, tol=0.001)

# ---- dynamic map vs bnb create_dynamic_map (runtime cross-check, [source] functional.py:296-348)
sys.path.insert(0, HERE)
from quant_ops import dynamic_map as our_dynamic_map  # noqa: E402

bnb = os.path.join(HERE, "..", "bitsandbytes")
sys.path.insert(0, bnb)
try:
    import bitsandbytes.functional as bF  # noqa: E402

    for signed in (True, False):
        ours = our_dynamic_map(signed=signed)
        theirs = bF.create_dynamic_map(signed=signed).flatten()
        same = bool(torch.allclose(ours.to(torch.float64), theirs.to(torch.float64), atol=1e-9))
        checks.append({"check": f"dynamic_map(signed={signed}) == bnb create_dynamic_map, "
                      "256 levels exact", "doc_value": True, "json_value": same, "pass": same})
        nz = int((ours != 0).sum().item())
        checks.append({"check": f"dynamic_map(signed={signed}) nonzero levels (255)",
                      "doc_value": 255, "json_value": nz, "pass": nz == 255})
except Exception as exc:  # pragma: no cover
    checks.append({"check": "dynamic_map vs bnb cross-check", "doc_value": True,
                   "json_value": f"error: {type(exc).__name__}", "pass": False})

# ---- gates summary: exactly one honest FAIL (G2) ----------------------------
check_eq("gates: exactly one FAIL (G2)", ["g2"],
         [k for k, v in {"g1": e1["gates"]["g1_nf4_beats_int4_on_N01"],
                         "g2": e2["gates"]["g2_int8_kernel_ge_1p2x_fp16"],
                         "g3": e3["gates"]["g3_loss_diff_lt_0.05"],
                         "g5": e5["gates"]["g5_consistency_within_10pct"]}.items()
          if not v])

all_pass = all(c["pass"] for c in checks)
out = {"exp": "doc_check", "n_checks": len(checks), "all_pass": all_pass,
       "n_fail": sum(1 for c in checks if not c["pass"]), "checks": checks}
with open(os.path.join(RES, "doc_check.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(f"doc_check: {len(checks)} checks, {out['n_fail']} fail, all_pass={all_pass}")
if not all_pass:
    for c in checks:
        if not c["pass"]:
            print("FAIL:", c["check"], "doc=", c["doc_value"], "json=", c["json_value"])
    raise SystemExit(1)
