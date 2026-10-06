"""AR004 triton-lab: three-runtime benchmark framework (triton / torch-cuBLAS / numba-venv).

Protocol (shared with AR002/003, cross-AR comparable):
- session burn-in 2s FP32 big GEMM, per-config 3 warmup, adaptive reps median, CUDA events
- correctness gates BEFORE timing (see srs 3.1)
- JSON per experiment (whole-exp overwrite semantics: collect records, save once)
- numba leg runs in the AR001 venv python as a back-to-back subprocess (same session clock)

Usage:
  python bench_dsl.py --exp E1|E2|E3|E4|all|probe
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

import kernels_triton as kt

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
NUMBA_VENV_PY = r"D:\Infra\01-foundations\GPU-Puzzles\solutions\.venv\Scripts\python.exe"
NUMBA_REF = os.path.join(HERE, "numba_ref.py")
HBM_PEAK = 448.0      # GB/s, Quadro RTX 5000
FP32_PEAK = 11.2      # TFLOPS
FP16_TC_PEAK = 89.2   # TFLOPS
L2_BYTES = 4 * 1024 * 1024


def env_snapshot():
    import triton
    p = torch.cuda.get_device_properties(0)
    return {"triton": triton.__version__, "triton_path": os.path.dirname(triton.__file__),
            "torch": torch.__version__, "gpu": p.name, "cc": f"{p.major}.{p.minor}",
            "sm_count": p.multi_processor_count, "total_mem_gb": round(p.total_memory / 2**30, 1),
            "shared_per_block_kb": 64}


def save(exp_name, records, extra=None):
    doc = {"exp": exp_name, "env": ENV, "records": records}
    if extra:
        doc.update(extra)
    path = os.path.join(RESULTS, exp_name.split("_")[0].lower() + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"saved -> {path} [{exp_name}] {len(records)} records")


def gate(name, ok):
    if not ok:
        raise RuntimeError(f"GATE FAIL: {name}")
    print(f"[gate] {name}: PASS")


def burn(seconds=2.0):
    a = torch.randn(2048, 2048, device=DEV)
    b = torch.randn(2048, 2048, device=DEV)
    t0 = time.time()
    while time.time() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def median_time_ms(fn, target_seconds=1.0, min_reps=3, max_reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); fn(); e.record(); e.synchronize()
    t1 = s.elapsed_time(e) / 1e3
    reps = max(min_reps, min(max_reps, int(round(target_seconds / max(t1, 1e-9)))))
    times = []
    for _ in range(reps):
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], reps


def ptx_counts(handle):
    ptx = handle.asm["ptx"]
    return {"mma_count": ptx.count("mma.sync"), "ldmatrix_count": ptx.count("ldmatrix"),
            "fma_f32_count": ptx.count("fma.rn.f32")}


def clock_now():
    """SM clock telemetry via nvidia-smi (evidence for clock-state protocol)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as ex:
        return f"unavailable: {type(ex).__name__}"


# ------------------------------------------------------------------ E1 add
def exp_e1():
    sizes = [2 ** i for i in range(12, 27)]
    # PHASE A: pre-compile all triton variants (no timing while compiling)
    xa = torch.rand(4096, device=DEV); ya = torch.rand(4096, device=DEV)
    oa = torch.empty_like(xa)
    for bs in (256, 1024, 4096):
        kt.add(xa, ya, oa, BLOCK_SIZE=bs)
    torch.cuda.synchronize()
    print(f"[E1] pre-compiled; clock={clock_now()}")
    # PHASE B: burn, then time everything back-to-back
    burn(2.0)
    print(f"[E1] burned; clock={clock_now()}")
    records = []
    for n in sizes:
        x = torch.rand(n, device=DEV)
        y = torch.rand(n, device=DEV)
        out = torch.empty_like(x)
        ref = x + y
        gb = lambda ms: 3 * n * 4 / (ms * 1e-3) / 1e9
        # torch
        torch.add(x, y, out=out)
        torch.cuda.synchronize()
        gate(f"add n={n} torch exact", torch.equal(out, ref))
        ms, reps = median_time_ms(lambda: torch.add(x, y, out=out))
        records.append({"variant": "torch", "n": n, "time_ms": round(ms, 6),
                        "gbps": round(gb(ms), 2), "reps": reps})
        # triton blocks
        for bs in (256, 1024, 4096):
            out.zero_()
            kt.add(x, y, out, BLOCK_SIZE=bs)
            torch.cuda.synchronize()
            gate(f"add n={n} triton bs={bs} exact", torch.equal(out, ref))
            ms, reps = median_time_ms(lambda: kt.add(x, y, out, BLOCK_SIZE=bs))
            records.append({"variant": f"triton_bs{bs}", "n": n, "time_ms": round(ms, 6),
                            "gbps": round(gb(ms), 2), "reps": reps})
        print(f"E1 n={n}: torch {records[-4]['gbps']:.1f} GB/s, "
              f"triton1024 {records[-2]['gbps']:.1f} GB/s")
    save("E1", records, extra={"clock_after": clock_now()})
    # numba leg (same session, back-to-back subprocess)
    run_numba("add", sizes, os.path.join(RESULTS, "e1na.json"))


def run_numba(exp, sizes, out_path):
    if not os.path.exists(NUMBA_VENV_PY):
        print(f"[numba] venv python missing: {NUMBA_VENV_PY} — SKIPPED")
        return None
    cmd = [NUMBA_VENV_PY, NUMBA_REF, "--exp", exp, "--out", out_path, "--sizes"] + \
        [str(s) for s in sizes]
    print(f"[numba] running: {exp} sizes={len(sizes)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    print(r.stdout.strip()[-1500:] if r.stdout else "")
    if r.returncode != 0:
        print(f"[numba] STDERR: {r.stderr.strip()[-1000:]}")
        raise RuntimeError("numba subprocess failed")
    with open(out_path, encoding="utf-8") as f:
        doc = json.load(f)
    doc["same_session"] = True
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[numba] merged in {time.time() - t0:.1f}s ({len(doc['records'])} records)")
    return doc


# ------------------------------------------------------------------ E2 softmax
def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    num = torch.exp(z)
    den = num.sum(dim=1)
    return num / den[:, None]


def exp_e2():
    total = 2 ** 20
    widths = (1024, 2048, 4096, 8192, 16384)
    # PHASE A: pre-compile triton kernel per (width, num_warps) — BLOCK_SIZE is constexpr
    for ncols in widths:
        m = total // ncols
        x = torch.randn(m, ncols, device=DEV)
        y = torch.empty_like(x)
        for nw in (4, 8):
            kt.softmax(x, y, num_warps=nw)
    torch.cuda.synchronize()
    print(f"[E2] pre-compiled; clock={clock_now()}")
    # PHASE B: burn then time back-to-back
    burn(2.0)
    print(f"[E2] burned; clock={clock_now()}")
    records = []
    for ncols in widths:
        m = total // ncols
        x = torch.randn(m, ncols, device=DEV)
        # extreme-value rows for numerical stability gate (first row scaled)
        x[0, ::2] = 1000.0
        x[0, 1::2] = -1000.0
        ref64 = torch.softmax(x.double(), dim=1)
        eff_bytes = 2 * m * ncols * 4
        naive_bytes = (8 * m * ncols + 4 * m) * 4
        y = torch.empty_like(x)
        # triton (num_warps 4 and 8)
        for nw in (4, 8):
            kt.softmax(x, y, num_warps=nw)
            torch.cuda.synchronize()
            err = (y.double() - ref64).abs().max().item()
            gate(f"softmax {m}x{ncols} triton nw={nw} vs double", err <= 1e-6)
            ms, reps = median_time_ms(lambda: kt.softmax(x, y, num_warps=nw))
            records.append({"variant": f"triton_nw{nw}", "M": m, "N": ncols,
                            "time_ms": round(ms, 6),
                            "gbps_eff": round(eff_bytes / (ms * 1e-3) / 1e9, 2),
                            "reps": reps})
        # torch native fused
        got = torch.nn.functional.softmax(x, dim=1)
        err = (got.double() - ref64).abs().max().item()
        gate(f"softmax {m}x{ncols} torch native vs double", err <= 1e-6)
        ms, reps = median_time_ms(lambda: torch.nn.functional.softmax(x, dim=1))
        records.append({"variant": "torch_native", "M": m, "N": ncols,
                        "time_ms": round(ms, 6),
                        "gbps_eff": round(eff_bytes / (ms * 1e-3) / 1e9, 2), "reps": reps})
        # naive 5-pass
        got = naive_softmax(x)
        err = (got.double() - ref64).abs().max().item()
        gate(f"softmax {m}x{ncols} naive vs double", err <= 1e-6)
        ms, reps = median_time_ms(lambda: naive_softmax(x))
        records.append({"variant": "naive_5pass", "M": m, "N": ncols,
                        "time_ms": round(ms, 6),
                        "gbps_eff": round(eff_bytes / (ms * 1e-3) / 1e9, 2),
                        "gbps_actual": round(naive_bytes / (ms * 1e-3) / 1e9, 2),
                        "reps": reps})
        print(f"E2 {m}x{ncols}: triton {records[-4]['gbps_eff']:.0f}/"
              f"{records[-3]['gbps_eff']:.0f} GB/s, native {records[-2]['gbps_eff']:.0f}, "
              f"naive {records[-1]['gbps_eff']:.0f}")
    save("E2", records, extra={"clock_after": clock_now()})


# ------------------------------------------------------------------ E3 matmul FP32
def config_grid(bms, bns, bks, warps_list, stages_list):
    for bm in bms:
        for bn in bns:
            for bk in bks:
                for w in warps_list:
                    for s in stages_list:
                        shared = s * bk * (bm + bn) * 4
                        if shared > 64 * 1024:
                            yield (bm, bn, bk, w, s, f"pruned: shared {shared//1024}KB > 64KB")
                        else:
                            yield (bm, bn, bk, w, s, None)


def bench_config(a, b, c, bm, bn, bk, w, s, group_m=8):
    fn = lambda: kt.matmul(a, b, c, BM=bm, BN=bn, BK=bk, GROUP_M=group_m,
                           num_warps=w, num_stages=s)
    fn()
    torch.cuda.synchronize()
    ms, reps = median_time_ms(fn)
    return ms, reps


def exp_e3():
    sizes = [256, 512, 1024, 2048]
    rng = torch.Generator(device=DEV).manual_seed(42)
    sweep_n = 1024
    valid_cfgs = [c for c in config_grid((32, 64, 128), (32, 64, 128), (16, 32, 64), (4, 8), (2, 3))
                  if c[5] is None]

    # ---- PHASE A: compile ALL valid configs once (N is runtime arg, cache shared across N)
    a = torch.randn(sweep_n, sweep_n, device=DEV, generator=rng)
    b = torch.randn(sweep_n, sweep_n, device=DEV, generator=rng)
    c = torch.empty_like(a)
    handles = {}
    for bm, bn, bk, w, s, _ in valid_cfgs:
        try:
            handles[(bm, bn, bk, w, s)] = kt.matmul(a, b, c, BM=bm, BN=bn, BK=bk,
                                                    GROUP_M=8, num_warps=w, num_stages=s)
            torch.cuda.synchronize()
        except Exception as ex:
            handles[(bm, bn, bk, w, s)] = f"{type(ex).__name__}: {str(ex)[:100]}"
    print(f"[E3] phase A: compiled {sum(1 for v in handles.values() if not isinstance(v, str))}"
          f"/{len(valid_cfgs)} configs; clock={clock_now()}")

    # ---- PHASE B: burn, then correctness+timing back-to-back (no compiles in loop)
    burn(2.0)
    print(f"[E3] burned; clock={clock_now()}")
    ref = a @ b
    cfg_records = []
    t0 = time.time()
    for bm, bn, bk, w, s, _ in valid_cfgs:
        h = handles[(bm, bn, bk, w, s)]
        rec = {"N": sweep_n, "BM": bm, "BN": bn, "BK": bk, "num_warps": w, "num_stages": s}
        if isinstance(h, str):
            rec["oor_reason"] = h
            cfg_records.append(rec)
            continue
        c.zero_()
        kt.matmul(a, b, c, BM=bm, BN=bn, BK=bk, GROUP_M=8, num_warps=w, num_stages=s)
        torch.cuda.synchronize()
        rel = ((c - ref).abs().max() / ref.abs().max()).item()
        gate(f"E3cfg {bm}x{bn}x{bk} w{w} s{s}", rel <= 1e-5)
        ms, reps = bench_config(a, b, c, bm, bn, bk, w, s)
        rec.update({"time_ms": round(ms, 6),
                    "tflops": round(2 * sweep_n ** 3 / (ms * 1e-3) / 1e12, 3),
                    "reps": reps})
        cfg_records.append(rec)
    # pruned configs (never attempted)
    for bm, bn, bk, w, s, why in config_grid((32, 64, 128), (32, 64, 128), (16, 32, 64), (4, 8), (2, 3)):
        if why:
            cfg_records.append({"N": sweep_n, "BM": bm, "BN": bn, "BK": bk,
                                "num_warps": w, "num_stages": s, "oor_reason": why})
    print(f"E3cfg sweep N={sweep_n}: {sum(1 for r in cfg_records if 'tflops' in r)} ok, "
          f"{sum(1 for r in cfg_records if 'oor_reason' in r)} pruned/failed, "
          f"{time.time() - t0:.0f}s; clock={clock_now()}")
    save("E3cfg", cfg_records)

    # ---- top-5 configs at other sizes (cache hit, fresh burn per size)
    ok_cfgs = sorted([r for r in cfg_records if "tflops" in r], key=lambda r: -r["tflops"])
    top = [(r["BM"], r["BN"], r["BK"], r["num_warps"], r["num_stages"]) for r in ok_cfgs[:5]]
    for n in [s for s in sizes if s != sweep_n]:
        burn(1.0)
        a = torch.randn(n, n, device=DEV, generator=rng)
        b = torch.randn(n, n, device=DEV, generator=rng)
        c = torch.empty_like(a)
        ref = a @ b
        for bm, bn, bk, w, s in top:
            try:
                c.zero_()
                kt.matmul(a, b, c, BM=bm, BN=bn, BK=bk, GROUP_M=8, num_warps=w, num_stages=s)
                torch.cuda.synchronize()
                rel = ((c - ref).abs().max() / ref.abs().max()).item()
                gate(f"E3cfg N={n} {bm}x{bn}x{bk} w{w} s{s}", rel <= 1e-5)
                ms, reps = bench_config(a, b, c, bm, bn, bk, w, s)
                cfg_records.append({"N": n, "BM": bm, "BN": bn, "BK": bk,
                                    "num_warps": w, "num_stages": s,
                                    "time_ms": round(ms, 6),
                                    "tflops": round(2 * n ** 3 / (ms * 1e-3) / 1e12, 3),
                                    "reps": reps})
            except Exception as ex:
                cfg_records.append({"N": n, "BM": bm, "BN": bn, "BK": bk,
                                    "num_warps": w, "num_stages": s,
                                    "oor_reason": f"{type(ex).__name__}: {str(ex)[:100]}"})
    save("E3cfg", cfg_records)

    # ---- three-way table + GROUP_M effect (fresh burn per N)
    records = []
    for n in sizes:
        burn(1.0)
        a = torch.randn(n, n, device=DEV, generator=rng)
        b = torch.randn(n, n, device=DEV, generator=rng)
        c = torch.empty_like(a)
        best = max([r for r in cfg_records if r.get("N") == n and "tflops" in r],
                   key=lambda r: r["tflops"])
        bm, bn, bk, w, s = best["BM"], best["BN"], best["BK"], best["num_warps"], best["num_stages"]
        ref = a @ b
        for gm in (1, 8):
            fn = lambda: kt.matmul(a, b, c, BM=bm, BN=bn, BK=bk, GROUP_M=gm,
                                   num_warps=w, num_stages=s)
            fn(); torch.cuda.synchronize()
            rel = ((c - ref).abs().max() / ref.abs().max()).item()
            gate(f"E3 N={n} triton gm={gm}", rel <= 1e-5)
            ms, reps = median_time_ms(fn)
            records.append({"N": n, "variant": f"triton_best_gm{gm}",
                            "config": f"{bm}x{bn}x{bk}_w{w}_s{s}_G{gm}",
                            "time_ms": round(ms, 6),
                            "tflops": round(2 * n ** 3 / (ms * 1e-3) / 1e12, 3), "reps": reps})
        ms, reps = median_time_ms(lambda: torch.matmul(a, b, out=c))
        records.append({"N": n, "variant": "cublas", "config": "heuristic",
                        "time_ms": round(ms, 6),
                        "tflops": round(2 * n ** 3 / (ms * 1e-3) / 1e12, 3), "reps": reps})
        print(f"E3 N={n}: triton {records[-3]['tflops']:.2f} TF (cfg {records[-3]['config']}), "
              f"cuBLAS {records[-1]['tflops']:.2f} TF; clock={clock_now()}")
    save("E3", records)
    run_numba("gemm", sizes, os.path.join(RESULTS, "e3na.json"))


# ------------------------------------------------------------------ E4 matmul FP16
def exp_e4():
    sizes = [512, 1024, 2048]
    rng = torch.Generator(device=DEV).manual_seed(42)
    valid = [c for c in config_grid((64, 128), (64, 128), (16, 32), (4, 8), (2, 3))
             if c[5] is None]
    # PHASE A: compile all fp16 configs at N=512 (cache shared across N)
    a = torch.randn(512, 512, device=DEV, dtype=torch.float16, generator=rng)
    b = torch.randn(512, 512, device=DEV, dtype=torch.float16, generator=rng)
    c = torch.empty_like(a)
    handles = {}
    for bm, bn, bk, w, s, _ in valid:
        try:
            handles[(bm, bn, bk, w, s)] = kt.matmul(a, b, c, BM=bm, BN=bn, BK=bk, GROUP_M=8,
                                                    num_warps=w, num_stages=s)
            torch.cuda.synchronize()
        except Exception as ex:
            handles[(bm, bn, bk, w, s)] = f"{type(ex).__name__}: {str(ex)[:100]}"
    print(f"[E4] phase A: compiled {sum(1 for v in handles.values() if not isinstance(v, str))}"
          f"/{len(valid)} fp16 configs; clock={clock_now()}")

    records = []
    ptx_summary = []
    for n in sizes:
        burn(1.0)
        a = torch.randn(n, n, device=DEV, dtype=torch.float16, generator=rng)
        b = torch.randn(n, n, device=DEV, dtype=torch.float16, generator=rng)
        c = torch.empty_like(a)
        ref = (a.float() @ b.float()).half()
        best = None
        for bm, bn, bk, w, s, _ in valid:
            h = handles[(bm, bn, bk, w, s)]
            rec = {"N": n, "BM": bm, "BN": bn, "BK": bk, "num_warps": w, "num_stages": s}
            if isinstance(h, str):
                rec["oor_reason"] = h
                records.append(rec)
                continue
            c.zero_()
            handle = kt.matmul(a, b, c, BM=bm, BN=bn, BK=bk, GROUP_M=8,
                               num_warps=w, num_stages=s)
            torch.cuda.synchronize()
            err = (c.float() - ref.float()).abs().max().item()
            # 1.5 ulp@max|ref| gate (AR003 E5 precedent): fp16 output rounding is
            # ~1-2 ulp at the result magnitude by construction
            max_ref = ref.float().abs().max().item()
            ulp = 2.0 ** (math.floor(math.log2(max(max_ref, 1e-30))) - 10)
            gate(f"E4 N={n} {bm}x{bn}x{bk} w{w} s{s}", err <= 1.5 * ulp)
            ms, reps = bench_config(a, b, c, bm, bn, bk, w, s)
            counts = ptx_counts(handle)
            rec.update({"time_ms": round(ms, 6),
                        "tflops": round(2 * n ** 3 / (ms * 1e-3) / 1e12, 3),
                        "reps": reps, "n_regs": handle.n_regs,
                        "n_spills": handle.n_spills, **counts})
            records.append(rec)
            if best is None or rec["tflops"] > best["tflops"]:
                best = rec
        # cuBLAS fp16
        ms, reps = median_time_ms(lambda: torch.matmul(a, b, out=c))
        records.append({"N": n, "variant": "cublas_fp16", "time_ms": round(ms, 6),
                        "tflops": round(2 * n ** 3 / (ms * 1e-3) / 1e12, 3), "reps": reps})
        if best:
            ptx_summary.append({"N": n, "best_config": best})
            print(f"E4 N={n}: triton best {best['tflops']:.2f} TF "
                  f"({best['BM']}x{best['BN']}x{best['BK']} w{best['num_warps']} "
                  f"s{best['num_stages']}, mma={best['mma_count']}, spills={best['n_spills']}), "
                  f"cuBLAS {records[-1]['tflops']:.2f} TF; clock={clock_now()}")
    save("E4", records, extra={"ptx_best": ptx_summary})


# ------------------------------------------------------------------ main
def main():
    global ENV
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="E1, E2, E3, E4, all, probe")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs(RESULTS, exist_ok=True)

    if args.exp == "probe":
        ENV = env_snapshot()
        probe = dict(ENV)
        if os.path.exists(NUMBA_VENV_PY):
            r = subprocess.run([NUMBA_VENV_PY, NUMBA_REF, "--exp", "probe"],
                               capture_output=True, text=True, cwd=HERE)
            probe["numba_venv"] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else f"probe failed: {r.stderr[-200:]}"
        else:
            probe["numba_venv"] = "missing"
        print(json.dumps(probe, indent=2))
        with open(os.path.join(RESULTS, "env.json"), "w", encoding="utf-8") as f:
            json.dump(probe, f, indent=2)
        return

    ENV = env_snapshot()
    print("env:", ENV)
    print("burning ~2s for clock ramp...")
    burn(2.0)

    table = {"E1": exp_e1, "E2": exp_e2, "E3": exp_e3, "E4": exp_e4}
    exps = ["E1", "E2", "E3", "E4"] if args.exp == "all" else args.exp.split("+")
    for name in exps:
        t0 = time.time()
        table[name]()
        print(f"[{name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
