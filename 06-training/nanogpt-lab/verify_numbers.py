"""AR006: assert that every number quoted in results.md / notes matches results/*.json.

Writes results/doc_check.json (all_pass flag + per-check detail).
Mirrors AR005's --verify-cache lesson: document numbers must have code-level evidence.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


e1, e2, e3, e4, e5 = (load("e1.json"), load("e2.json"), load("e3.json"),
                      load("e4.json"), load("e5.json"))
corpus = load(os.path.join("..", "data", "corpus_stats.json").replace("\\", os.sep)
              if False else os.path.join(HERE, "data", "corpus_stats.json"))

checks = []


def check(name, doc, actual, tol=0.005):
    ok = (doc is not None and actual is not None
          and abs(doc - actual) <= tol * max(1.0, abs(actual)))
    checks.append({"check": name, "doc_value": doc, "json_value": actual, "pass": bool(ok)})
    return ok


# corpus
check("corpus total chars", 898703, corpus["total_chars"])
check("corpus vocab", 124, corpus["vocab_size"])
check("corpus entropy bits/char", 5.162, corpus["char_entropy_bits"], tol=0.001)
check("train chars", 853768, corpus["train_chars"])
check("val chars", 44935, corpus["val_chars"])

# E1
check("e1 final train loss (0.085)", 0.085, e1["final_train_loss"], tol=0.01)
vc = e1["val_curve"]
imin = vc["vals"].index(min(vc["vals"]))
check("e1 val min (1.468)", 1.468, vc["vals"][imin], tol=0.002)
check("e1 val min step (750)", 750, vc["steps"][imin], tol=0)
check("e1 final val (2.348)", 2.348, e1["final_val_loss"], tol=0.002)
check("e1 tok/s (300163)", 300163, e1["tokens_per_s"], tol=0.002)
check("e1 ms/step (54.6)", 54.6, e1["step_time_ms_median"], tol=0.01)
check("e1 MFU pct (23.9)", 23.9, e1["mfu_pct"], tol=0.01)
check("e1 scaler init (65536)", 65536, e1["scaler"]["init"], tol=0)
check("e1 scaler final (131072)", 131072, e1["scaler"]["final"], tol=0)
check("e1 scaler backoffs (0)", 0, e1["scaler"]["backoffs"], tol=0)
check("e1 params non-embed (10669440)", 10669440, e1["n_params_non_embed"], tol=0)
epochs = e1["steps_run"] * e1["batch"] * e1["block"] / corpus["train_chars"]
check("e1 epochs (~58)", 58, epochs, tol=0.02)

# E2
for k, tok, ms, fin, mfu, mtol in (("fp32", 255305, 64.2, 0.727, 50.7, 0.01),
                                    ("fp16", 682175, 24.0, 0.729, 17.0, 0.01),
                                    ("bf16", 146872, 111.6, 0.728, 3.7, 0.02)):
    check(f"e2 {k} tok/s ({tok})", tok, e2[k]["tokens_per_s"], tol=0.002)
    check(f"e2 {k} ms/step ({ms})", ms, e2[k]["step_time_ms_median"], tol=0.01)
    check(f"e2 {k} final loss ({fin})", fin, e2[k]["final_train_loss"], tol=0.003)
    check(f"e2 {k} MFU ({mfu})", mfu, e2[k]["mfu_pct"], tol=mtol)
check("e2 fp16 speedup (2.67)", 2.67, e2["fp16_vs_fp32_speedup"], tol=0.005)
check("e2 bf16/fp32 ratio (0.575)", 0.575,
      e2["bf16"]["tokens_per_s"] / e2["fp32"]["tokens_per_s"], tol=0.01)

# E3
sizes = ["0.5M", "3M", "10.5M", "25M"]
for s, v, m, t, p in zip(sizes, (1.755, 1.576, 1.912, 2.063),
                         (7.5, 17.1, 23.7, 29.9),
                         (2065063, 686045, 297365, 162490),
                         (409728, 3179776, 10669440, 25238016)):
    check(f"e3 {s} val ({v})", v, e3["sizes"][s]["final_val_loss"], tol=0.003)
    check(f"e3 {s} MFU ({m})", m, e3["sizes"][s]["mfu_pct"], tol=0.01)
    check(f"e3 {s} tok/s ({t})", t, e3["sizes"][s]["tokens_per_s"], tol=0.002)
    check(f"e3 {s} non-embed params ({p})", p, e3["sizes"][s]["n_params_non_embed"], tol=0)
sp = e3["step_time_pred_from_25M"]
check("e3 10.5M pred ms (43.7)", 43.7, sp["10.5M_pred_ms"], tol=0.005)
check("e3 10.5M meas ms (55.1)", 55.1, sp["10.5M_meas_ms"], tol=0.005)
check("e3 pred ratio (1.26)", 1.26, sp["ratio_meas_over_pred"], tol=0.01)
# derived linear-in-flops predictions for 0.5M/3M (same formula as gpt_lab e3(),
# computed from e3.json fields; quoted in results.md E3 conclusion 3)
t25 = e3["sizes"]["25M"]["step_time_ms_median"]
n25 = e3["sizes"]["25M"]["flops_per_token"]
pred05 = t25 * e3["sizes"]["0.5M"]["flops_per_token"] / n25
pred3 = t25 * e3["sizes"]["3M"]["flops_per_token"] / n25
check("e3 0.5M pred ms (~2.0)", 2.0, pred05, tol=0.01)
check("e3 3M pred ms (~13.7)", 13.7, pred3, tol=0.01)
check("e3 0.5M meas/pred (3.98)", 3.98,
      e3["sizes"]["0.5M"]["step_time_ms_median"] / pred05, tol=0.01)
check("e3 3M meas/pred (1.75)", 1.75,
      e3["sizes"]["3M"]["step_time_ms_median"] / pred3, tol=0.01)
# E1 first val point (results.md quotes 2.193)
check("e1 first val (2.193)", 2.193, e1["val_curve"]["vals"][0], tol=0.002)
# E2 VRAM peaks (results.md table)
for k, mb in (("fp32", 1225), ("fp16", 774), ("bf16", 1226)):
    check(f"e2 {k} vram peak MB ({mb})", mb, e2[k]["vram_peak_mb"], tol=0.005)

# E4
lr_doc = {"lr0.0001_cosine": 2.424, "lr0.0001_constant": 2.107,
          "lr0.0003_cosine": 1.840, "lr0.0003_constant": 1.498,
          "lr0.001_cosine": 1.109, "lr0.001_constant": 1.025,
          "lr0.003_cosine": 0.822, "lr0.003_constant": 0.972}
for tag, v in lr_doc.items():
    check(f"e4 {tag} ({v})", v, e4["lr_sweep"][tag]["final_train_loss"], tol=0.003)
for b, st, tok, lo in (("B16", 3198, 364884, 0.706),
                       ("B64", 800, 685883, 1.102),
                       ("B256", 200, 768968, 2.411)):
    check(f"e4 {b} steps ({st})", st, e4["batch_sweep"][b]["steps"], tol=0)
    check(f"e4 {b} tok/s ({tok})", tok, e4["batch_sweep"][b]["tokens_per_s"], tol=0.002)
    check(f"e4 {b} final loss ({lo})", lo, e4["batch_sweep"][b]["final_train_loss"], tol=0.003)

# E5
check("e5 linear/matmul pct (27.0)", 27.0, e5["categories"]["linear/matmul"]["pct"], tol=0.005)
check("e5 top5 pct sum (50.9)", 50.9, sum(o["pct"] for o in e5["top_ops"][:5]), tol=0.005)
check("e5 elementwise/copy (11.3)", 11.3, e5["categories"]["elementwise/copy"]["pct"], tol=0.01)
check("e5 attention (9.9)", 9.9, e5["categories"]["attention"]["pct"], tol=0.01)
check("e5 layernorm (9.5)", 9.5, e5["categories"]["layernorm"]["pct"], tol=0.01)
check("e5 eager overhead (9.1)", 9.1, e5["categories"]["eager_overhead"]["pct"], tol=0.01)
check("e5 activation (5.9)", 5.9, e5["categories"]["activation"]["pct"], tol=0.01)
check("e5 optimizer (2.4)", 2.4, e5["categories"]["optimizer"]["pct"], tol=0.01)
tr = [o for o in e5["top_ops"] if o["name"] == "aten::transpose"][0]
check("e5 transpose count (4900)", 4900, tr["count"], tol=0)
mm = [o for o in e5["top_ops"] if o["name"] == "aten::mm"][0]
check("e5 mm count (1500)", 1500, mm["count"], tol=0)
check("e5 mm pct (25.7)", 25.7, mm["pct"], tol=0.005)
check("e5 fresh tok/s (301651)", 301651, e5["fresh_tokens_per_s"], tol=0.002)
check("e5 fresh vs e1 within 10%", 1.0,
      1 if abs(e5["fresh_tokens_per_s"] / e5["e1_tokens_per_s"] - 1) < 0.10 else 0, tol=0)

# E7
e7 = load("e7.json")
e7c = load("e7_compile.json")
check("e7 fp16 flash (0= False)", 0, 1 if e7["fp16"]["flash"] else 0, tol=0)
check("e7 fp16 mem_efficient (1= True)", 1, 1 if e7["fp16"]["mem_efficient"] else 0, tol=0)
check("e7 fp32 flash (0= False)", 0, 1 if e7["fp32"]["flash"] else 0, tol=0)
check("e7 compile failed (1)", 1, 1 if e7c["status"] == "failed" else 0, tol=0)

out = {"exp": "doc_number_check", "n_checks": len(checks),
       "n_pass": sum(1 for c in checks if c["pass"]),
       "all_pass": all(c["pass"] for c in checks), "checks": checks}
with open(os.path.join(RES, "doc_check.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
for c in checks:
    if not c["pass"]:
        print("FAIL:", c["check"], "doc", c["doc_value"], "json", c["json_value"])
print(f"{out['n_pass']}/{out['n_checks']} checks pass, all_pass={out['all_pass']}")
assert out["all_pass"]
