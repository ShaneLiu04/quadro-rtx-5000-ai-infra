import argparse
import json
import os
import statistics
import time

import torch

DEV = "cuda"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Machine constants (RTX 5000 spec sheet / AR002 env snapshot)
FP32_PEAK_TFLOPS = 11.2
FP16_PEAK_TFLOPS = 89.2
INT8_PEAK_TOPS = 178.4
HBM_GBPS = 448.0
L2_MB = 4.0
NUM_SM = 48


def burn(seconds=2.0):
    a = torch.randn(2048, 2048, device=DEV)
    t0 = time.time()
    while time.time() - t0 < seconds:
        a @ a
    torch.cuda.synchronize()
    del a


def median_time_ms(fn, target_seconds=1.0, warmup=3, min_reps=3, max_reps=10, passes=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    t1 = start.elapsed_time(end)
    reps = max(min_reps, min(max_reps, round(target_seconds * 1000.0 / max(t1, 1e-3))))
    pass_times = []
    for _ in range(passes):
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        pass_times.append(start.elapsed_time(end) / reps)
    return statistics.median(pass_times), reps, pass_times


def per_iter_time_ms(fn, reps=5, warmup=3, passes=3, prep=None):
    for _ in range(warmup):
        if prep is not None:
            prep()
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    pass_times = []
    for _ in range(passes):
        samples = []
        for _ in range(reps):
            if prep is not None:
                prep()
                torch.cuda.synchronize()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        pass_times.append(statistics.median(samples))
    return statistics.median(pass_times), reps, pass_times


def env_snapshot():
    prop = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "gpu": prop.name,
        "cc": f"{prop.major}.{prop.minor}",
        "sm_count": prop.multi_processor_count,
        "total_mem_gb": round(prop.total_memory / 2**30, 1),
    }


def save(exp_name, records, extra=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"exp": exp_name, "env": env_snapshot(), "records": records}
    if extra:
        payload["extra"] = extra
    path = os.path.join(RESULTS_DIR, f"{exp_name.split('_')[0].lower()}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {"env": payload["env"], "experiments": {}}
    doc["experiments"][exp_name] = payload
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"saved -> {path} [{exp_name}] {len(records)} records")


def tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def gate(label, ok):
    status = "PASS" if ok else "FAIL"
    print(f"[gate] {label}: {status}")
    if not ok:
        raise AssertionError(f"correctness gate failed: {label}")


def probe_int_mm():
    info = {"available": False}
    if not hasattr(torch, "_int_mm"):
        info["error"] = "torch._int_mm not present"
        return info
    try:
        n = 128
        a = torch.randint(-4, 5, (n, n), dtype=torch.int8, device=DEV)
        b = torch.randint(-4, 5, (n, n), dtype=torch.int8, device=DEV)
        c = torch._int_mm(a, b)
        ref = (a.int() @ b.int())
        assert c.dtype == torch.int32
        assert torch.equal(c, ref)
        info["available"] = True
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def exp_e1a():
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    k = 2048
    records = []
    for m in sizes:
        a = torch.randn(m, k, device=DEV)
        for n in sizes:
            b = torch.randn(k, n, device=DEV)
            ms, reps, _ = median_time_ms(lambda: a @ b)
            waves = ((m + 127) // 128) * ((n + 127) // 128) / NUM_SM
            records.append({"M": m, "N": n, "K": k, "dtype": "fp32", "time_ms": round(ms, 6),
                            "tflops": round(tflops(m, n, k, ms), 3), "reps": reps,
                            "model_tiles": ((m + 127) // 128) * ((n + 127) // 128),
                            "model_waves": round(waves, 3)})
            print(f"E1a M={m} N={n}: {ms:.4f} ms, {tflops(m, n, k, ms):.2f} TFLOPS")
        del a
    save("E1a", records)


def exp_e1b():
    n, k = 4096, 4096
    ms_list = [64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1024, 1536, 2048]
    records = []
    b = torch.randn(k, n, device=DEV)
    for m in ms_list:
        a = torch.randn(m, k, device=DEV)
        ms, reps, _ = median_time_ms(lambda: a @ b)
        tiles = ((m + 127) // 128) * ((n + 127) // 128)
        records.append({"M": m, "N": n, "K": k, "dtype": "fp32", "time_ms": round(ms, 6),
                        "tflops": round(tflops(m, n, k, ms), 3), "reps": reps,
                        "model_tiles": tiles, "model_waves": round(tiles / NUM_SM, 3)})
        print(f"E1b M={m}: {ms:.4f} ms, {tflops(m, n, k, ms):.2f} TFLOPS, tiles={tiles}")
    save("E1b", records)


def exp_e2():
    n, k = 4096, 4096
    m_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    records = []
    b = torch.randn(k, n, device=DEV)
    ref_b = b.cpu().double()
    for m in m_list:
        a = torch.randn(m, k, device=DEV)
        if m <= 16:
            ref = (a.cpu().double() @ ref_b)
            got = (a @ b).cpu().double()
            gate(f"E2 m={m} correctness", torch.allclose(ref, got, atol=1e-3, rtol=1e-3))
        ms, reps, _ = median_time_ms(lambda: a @ b, min_reps=5)
        bytes_moved = 4 * (m * k + k * n + m * n)
        records.append({"M": m, "N": n, "K": k, "dtype": "fp32", "time_ms": round(ms, 6),
                        "tflops": round(tflops(m, n, k, ms), 3),
                        "gbps": round(bytes_moved / (ms * 1e-3) / 1e9, 1), "reps": reps})
        print(f"E2 M={m}: {ms:.4f} ms, {tflops(m, n, k, ms):.3f} TFLOPS, "
              f"{bytes_moved / (ms * 1e-3) / 1e9:.1f} GB/s")
    save("E2", records)


def exp_e3():
    m, n, k = 256, 256, 16384
    splits = [1, 2, 4, 8, 16]
    a = torch.randn(m, k, device=DEV)
    b = torch.randn(k, n, device=DEV)
    ref = a @ b

    c_parts = {s: [torch.empty(m, n, device=DEV) for _ in range(s)] for s in splits if s > 1}
    c_out = torch.empty(m, n, device=DEV)
    streams = [torch.cuda.Stream() for _ in range(splits[-1])]
    prev_done = None

    def chunk_views(s):
        acs = [a[:, i * k // s:(i + 1) * k // s] for i in range(s)]
        bcs = [b[i * k // s:(i + 1) * k // s, :] for i in range(s)]
        return acs, bcs

    def make_split_fn(s):
        acs, bcs = chunk_views(s)

        def fn():
            nonlocal prev_done
            cur = torch.cuda.current_stream()
            if prev_done is not None:
                for st in streams:
                    st.wait_event(prev_done)
            evs = []
            for i in range(s):
                st = streams[i]
                with torch.cuda.stream(st):
                    torch.matmul(acs[i], bcs[i], out=c_parts[s][i])
                    e = torch.cuda.Event()
                    e.record(st)
                    evs.append(e)
            for e in evs:
                cur.wait_event(e)
            c_out.copy_(c_parts[s][0])
            for i in range(1, s):
                c_out.add_(c_parts[s][i])
            done = torch.cuda.Event()
            done.record(cur)
            prev_done = done
        return fn

    def single_fn():
        torch.matmul(a, b, out=c_out)

    single_fn()
    torch.cuda.synchronize()
    gate("E3 s=1 correctness", torch.allclose(c_out, ref, atol=1e-2, rtol=1e-2))
    for s in splits:
        if s > 1:
            acs, bcs = chunk_views(s)
            a_s = torch.cat(acs, dim=1)
            b_s = torch.cat(bcs, dim=0)
            gate(f"E3 s={s} chunk sum correctness", torch.allclose(a_s @ b_s, ref, atol=1e-2, rtol=1e-2))

    records = []
    for s in splits:
        fn = single_fn if s == 1 else make_split_fn(s)
        label = "stream-parallel" if s > 1 else "single"
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        gate(f"E3 s={s} output correctness", torch.allclose(c_out, ref, atol=1e-2, rtol=1e-2))
        ms, reps, _ = median_time_ms(fn, target_seconds=1.0, min_reps=5)
        records.append({"M": m, "N": n, "K": k, "split_s": s, "variant": label,
                        "time_ms": round(ms, 6), "tflops": round(tflops(m, n, k, ms), 3),
                        "reps": reps,
                        "baseline_tiles": ((m + 127) // 128) * ((n + 127) // 128)})
        print(f"E3 s={s} ({label}): {ms:.4f} ms, {tflops(m, n, k, ms):.3f} TFLOPS")
    save("E3", records, extra={"note": "s>1: K split into s chunks, each on own CUDA stream, "
                                        "partial C summed on default stream (includes sum cost)"})


def exp_e4():
    dims = 256
    batches = [1, 2, 4, 8, 16, 32, 64]
    records = []
    ref_single = torch.randn(dims, dims, device=DEV)
    for bs in batches:
        x = torch.randn(bs, dims, dims, device=DEV)
        wf = torch.randn(dims, dims, device=DEV)
        w_same = wf.unsqueeze(0).expand(bs, dims, dims)
        xf = x.reshape(bs * dims, dims)

        got = torch.bmm(x, w_same)
        flat = xf @ wf
        gate(f"E4 bmm-vs-flat b={bs} equivalence",
             torch.allclose(got.reshape(bs * dims, dims), flat, atol=1e-3, rtol=1e-3))

        ms_bmm, reps, _ = median_time_ms(lambda: torch.bmm(x, w_same))
        ms_flat, reps2, _ = median_time_ms(lambda: xf @ wf)
        flops = 2.0 * bs * dims * dims * dims
        records.append({"batch": bs, "dims": dims,
                        "bmm_time_ms": round(ms_bmm, 6), "bmm_tflops": round(flops / (ms_bmm * 1e-3) / 1e12, 3),
                        "flat_time_ms": round(ms_flat, 6), "flat_tflops": round(flops / (ms_flat * 1e-3) / 1e12, 3),
                        "bmm_reps": reps, "flat_reps": reps2,
                        "bmm_tiles": ((dims + 127) // 128) * ((dims + 127) // 128) * bs})
        print(f"E4 b={bs}: bmm {ms_bmm:.4f} ms ({flops / (ms_bmm * 1e-3) / 1e12:.2f} TF) | "
              f"flat {ms_flat:.4f} ms ({flops / (ms_flat * 1e-3) / 1e12:.2f} TF)")
    save("E4", records)


def exp_e5(int_info):
    sizes = [512, 1024, 2048, 4096]
    records = []
    for n in sizes:
        a = torch.randn(n, n, device=DEV)
        b = torch.randn(n, n, device=DEV)
        ah, bh = a.half(), b.half()
        ref64 = ah.double() @ bh.double()
        got_half = ah @ bh
        acc_err = (got_half.float() - ref64.float()).abs().max().item()
        max_ref = ref64.abs().max().item()
        bound = 1.5 * 2**-11 * max_ref
        gate(f"E5 half n={n} accumulation err {acc_err:.4f} < 1.5ulp({bound:.4f})", acc_err < bound)
        rec = {"N": n}
        ms, reps, _ = median_time_ms(lambda: a @ b)
        rec["fp32_time_ms"] = round(ms, 6)
        rec["fp32_tflops"] = round(tflops(n, n, n, ms), 3)
        rec["fp32_reps"] = reps
        ms, reps, _ = median_time_ms(lambda: ah @ bh)
        rec["fp16_time_ms"] = round(ms, 6)
        rec["fp16_tflops"] = round(tflops(n, n, n, ms), 3)
        rec["fp16_reps"] = reps
        if int_info["available"]:
            ai = torch.randint(-4, 5, (n, n), dtype=torch.int8, device=DEV)
            bi = torch.randint(-4, 5, (n, n), dtype=torch.int8, device=DEV)
            got = torch._int_mm(ai, bi)
            ref = ai.int() @ bi.int()
            gate(f"E5 int8 n={n} exact", torch.equal(got, ref))
            ms, reps, _ = median_time_ms(lambda: torch._int_mm(ai, bi))
            rec["int8_time_ms"] = round(ms, 6)
            rec["int8_tops"] = round(tflops(n, n, n, ms), 3)
            rec["int8_reps"] = reps
        records.append(rec)
        print(f"E5 N={n}: fp32 {rec['fp32_tflops']:.2f} TF | fp16 {rec['fp16_tflops']:.2f} TF"
              + (f" | int8 {rec['int8_tops']:.2f} TOPS" if int_info["available"] else ""))
    save("E5", records, extra={"int_mm": int_info})


def exp_e3seq():
    m, n, k = 256, 256, 16384
    splits = [2, 4, 8, 16]
    a = torch.randn(m, k, device=DEV)
    b = torch.randn(k, n, device=DEV)
    ref = a @ b
    c_out = torch.empty(m, n, device=DEV)

    def chunk_views(s):
        acs = [a[:, i * k // s:(i + 1) * k // s] for i in range(s)]
        bcs = [b[i * k // s:(i + 1) * k // s, :] for i in range(s)]
        return acs, bcs

    def make_seq_fn(s):
        acs, bcs = chunk_views(s)
        parts = [torch.empty(m, n, device=DEV) for _ in range(s)]

        def fn():
            for i in range(s):
                torch.matmul(acs[i], bcs[i], out=parts[i])
            c_out.copy_(parts[0])
            for i in range(1, s):
                c_out.add_(parts[i])
        return fn

    records = []
    for s in splits:
        fn = make_seq_fn(s)
        fn()
        torch.cuda.synchronize()
        gate(f"E3seq s={s} output correctness", torch.allclose(c_out, ref, atol=1e-2, rtol=1e-2))
        ms, reps, _ = median_time_ms(fn, target_seconds=1.0, min_reps=5)
        print(f"E3seq s={s} (sequential): {ms:.4f} ms, {tflops(m, n, k, ms):.3f} TFLOPS")
        records.append({"M": m, "N": n, "K": k, "split_s": s, "variant": "sequential",
                        "time_ms": round(ms, 6), "tflops": round(tflops(m, n, k, ms), 3),
                        "reps": reps})
    save("E3seq", records)


def exp_e6():
    sizes = [512, 1024, 2048, 4096]
    flush_buf = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
    records = []
    for n in sizes:
        a = torch.randn(n, n, device=DEV)
        b = torch.randn(n, n, device=DEV)
        out = torch.empty(n, n, device=DEV)
        working_set_mb = 3 * n * n * 4 / 2**20

        def flush():
            flush_buf.fill_(1.0)

        def gemm():
            torch.matmul(a, b, out=out)

        ms_flush, _, pt_flush = per_iter_time_ms(gemm, reps=5, prep=flush)
        ms_nof, _, pt_nof = per_iter_time_ms(gemm, reps=5)
        rec = {"N": n, "working_set_mb": round(working_set_mb, 2), "l2_mb": L2_MB,
               "flush_time_ms": round(ms_flush, 6), "flush_tflops": round(tflops(n, n, n, ms_flush), 3),
               "noflush_time_ms": round(ms_nof, 6), "noflush_tflops": round(tflops(n, n, n, ms_nof), 3),
               "flush_passes_ms": [round(x, 6) for x in pt_flush],
               "noflush_passes_ms": [round(x, 6) for x in pt_nof],
               "speedup_noflush_vs_flush": round(ms_flush / ms_nof, 4)}
        records.append(rec)
        print(f"E6 N={n} (ws={working_set_mb:.1f}MB): flush {ms_flush:.4f} ms vs no-flush {ms_nof:.4f} ms"
              f" -> x{ms_flush / ms_nof:.2f}")
    save("E6", records, extra={"flush_method": "64MB fp32 buffer fill before each timed iteration, "
                                               "per-iter events both variants"})


def exp_e7():
    n = 2048
    a = torch.randn(n, n, device=DEV)
    b = torch.randn(n, n, device=DEV)
    a_cm = a.t().contiguous()
    b_cm = b.t().contiguous()

    out = torch.empty(n, n, device=DEV)

    variants = {
        "AB": lambda: torch.matmul(a, b, out=out),
        "AtB": lambda: torch.matmul(a.t(), b, out=out),
        "ABt": lambda: torch.matmul(a, b.t(), out=out),
        "AtBt": lambda: torch.matmul(a.t(), b.t(), out=out),
        "colA_colB": lambda: torch.matmul(a_cm.t(), b_cm.t(), out=out),
        "colA_B": lambda: torch.matmul(a_cm.t(), b, out=out),
        "A_colB": lambda: torch.matmul(a, b_cm.t(), out=out),
    }

    torch.cuda.synchronize()
    mem0 = torch.cuda.memory_allocated()
    for name, fn in variants.items():
        fn()
    torch.cuda.synchronize()
    mem1 = torch.cuda.memory_allocated()
    gate("E7 no implicit copies", mem1 - mem0 == 0)

    ref_ab = a @ b
    gate("E7 colA_colB same math as AB",
         torch.allclose(torch.matmul(a_cm.t(), b_cm.t()), ref_ab, atol=1e-3, rtol=1e-3))
    gate("E7 colA_B same math as AB",
         torch.allclose(torch.matmul(a_cm.t(), b), ref_ab, atol=1e-3, rtol=1e-3))
    gate("E7 AtBt equals (B@A).t",
         torch.allclose(torch.matmul(a.t(), b.t()), (b @ a).t(), atol=1e-3, rtol=1e-3))

    records = []
    for name, fn in variants.items():
        ms, reps, _ = median_time_ms(fn)
        records.append({"variant": name, "N": n, "time_ms": round(ms, 6),
                        "tflops": round(tflops(n, n, n, ms), 3), "reps": reps})
        print(f"E7 {name}: {ms:.4f} ms, {tflops(n, n, n, ms):.2f} TFLOPS")
    save("E7", records, extra={"note": "AB/AtB/ABt/AtBt are 4 distinct products (same shape/flops) "
                                        "= cuBLAS op flag combos; colA/colB variants preserve math of AB "
                                        "with col-major-stored operands"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True,
                        help="E1a, E1b, E2, E3, E3seq, E4, E5, E6, E7, probes, all")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.init()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    env_path = os.path.join(RESULTS_DIR, "env.json")
    env = env_snapshot()
    int_info = probe_int_mm()
    env["int_mm"] = int_info
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print("env:", env)

    if args.exp == "probes":
        return
    print("burning ~2s for clock ramp...")
    burn(2.0)

    exps = args.exp.split("+")
    table = {"E1a": exp_e1a, "E1b": exp_e1b, "E2": exp_e2, "E3": exp_e3, "E3seq": exp_e3seq,
             "E4": exp_e4, "E6": exp_e6, "E7": exp_e7}
    if "all" in exps:
        exps = ["E1a", "E1b", "E2", "E3", "E3seq", "E4", "E5", "E6", "E7"]
    for name in exps:
        t0 = time.time()
        if name == "E5":
            exp_e5(int_info)
        else:
            table[name]()
        print(f"[{name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
