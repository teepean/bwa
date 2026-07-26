#!/usr/bin/env python3
"""Tune ptxas Advanced Controls for the gpualn DFS kernel with NVIDIA CompileIQ.

The kernel sits on a jagged codegen landscape (PROGRESS.md: maxrregcount 48 -> 6,445 r/s,
40 -> 5,243, default -> 9,934), which is exactly what CompileIQ's search over the undocumented
ptxas scheduler controls is for. Source-level tuning has plateaued, so this is the remaining
CUDA-13-specific lever (nothing else in 13.x targets sm_86).

Objective: GPU-kernel seconds on sub100k (lower is better), taken as the best of N runs to
suppress thermal drift. Hard correctness gate: any config whose .sai md5 differs from the golden
value scores INVALID, so a "fast" but wrong schedule can never win.

Run:  /home/teemu/sorsa/CompileIQ/.venv/bin/python cuda/tune_compileiq.py [--generations G]
      [--pool P] [--repeats R] [--scheme]
Apply: nvcc ... -Xptxas=--apply-controls=cuda/gpualn.acf
"""
import argparse, hashlib, os, re, subprocess
from uuid import uuid4

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF    = "/home/dnastorage/aDNApipeline/hs37d5.fa"
READS  = os.path.join(REPO, "test_data/sub100k.fq")
GOLDEN = "eecf35c15db452bfe0500ce0b9dbd723"       # sub100k .sai, == CPU bwa aln
# NOTE: tune the engine you intend to SHIP. An ACF tuned for k_dfs_warp2 gave 1.174x there but
# ~1.0x on k_dfs_scheme -- the schedules do not transfer between the two kernels. When tuning the
# scheme engine, use a read set with 0% fallback (L<=63) or the objective mixes both kernels.
OBJS   = ["bwtgap.o", "bwtaln.o", "bwaseqio.o", "bamlite.o", "bwase.o"]

ARGS = None


def build(acf, exe):
    """acf=None builds with the default schedule (ptxas rejects an empty controls file)."""
    cmd = ["nvcc", "-O3", "-std=c++14", "-arch=sm_86", "-DALN_GPU_MAIN", "-I."]
    if acf:
        cmd.append(f"-Xptxas=--apply-controls={acf}")
    cmd += ["cuda/aln_gpu.cu", *OBJS, "-o", exe, "-L.", "-lbwa", "-lm", "-lz", "-lpthread", "-lrt"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    return r.returncode == 0


def run_once(exe):
    """returns (kernel_seconds, md5) or (None, None)"""
    env = dict(os.environ, GPUALN_HISTO="1")
    if ARGS.scheme:
        env["GPUALN_SCHEME"] = "1"
    r = subprocess.run([exe, "-l", "1024", "-n", "0.01", "-o", "2", "-t", "16", REF, ARGS.reads],
                       capture_output=True, timeout=1200, env=env, cwd=REPO)
    if r.returncode != 0:
        return None, None
    m = re.search(r"GPU-kernel ([0-9.]+)s", r.stderr.decode(errors="replace"))
    return (float(m.group(1)) if m else None), hashlib.md5(r.stdout).hexdigest()


def objective(config) -> float:
    uid = uuid4().hex[:8]
    acf, exe = f"/tmp/ciq_{uid}.acf", f"/tmp/ciq_{uid}.bin"
    try:
        if config is None:
            acf = None
        else:
            save_compiler_config(acf, config)
        if not build(acf, exe):
            return INVALID_SCORE
        best = None
        for _ in range(ARGS.repeats):
            t, md5 = run_once(exe)
            if t is None or md5 != ARGS.golden:  # correctness gate
                return INVALID_SCORE
            best = t if best is None else min(best, t)
        return best
    except Exception:
        return INVALID_SCORE
    finally:
        for f in (acf, exe):
            if not f: continue
            try: os.remove(f)
            except OSError: pass


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2, help="runs per config; score is the best")
    ap.add_argument("--scheme", action="store_true", help="tune the search-scheme engine instead")
    ap.add_argument("--reads", default=READS)
    ap.add_argument("--golden", default=GOLDEN, help="expected .sai md5 for --reads")
    ap.add_argument("--out", default=os.path.join(REPO, "cuda/gpualn.acf"))
    ARGS = ap.parse_args()

    ver = re.search(r"release (\d+\.\d+),", subprocess.run(
        ["ptxas", "--version"], capture_output=True, text=True, check=True).stdout).group(1)
    print(f"[tune] ptxas {ver}, engine={'scheme' if ARGS.scheme else 'exact'}, "
          f"{ARGS.generations} generations x pool {ARGS.pool}, {ARGS.repeats} repeats")

    base = objective(None)                     # default schedule, same measurement path
    if not isinstance(base, float):
        print("[tune] baseline FAILED (build/run/md5) -- aborting"); return
    print(f"[tune] baseline (no controls): {base:.3f}s")

    tuner = Search(objective_function=objective,
                   search_space=PtxasSearchSpace(version=ver),
                   search_config=SearchConfiguration(problem_type="min",
                                                     generations=ARGS.generations,
                                                     pool_size=ARGS.pool))
    results = tuner.start(num_workers=1)       # serialise: one GPU, timings must not contend
    best = results.get_best_result()
    sc = best.get("score_1") if best else None
    if not isinstance(sc, (int, float)):
        print(f"[tune] no valid configuration found (best={sc!r}); keeping the default schedule")
        return
    print(f"[tune] best {sc:.3f}s vs baseline {base:.3f}s = {base/sc:.3f}x")
    if sc < base:
        save_compiler_config(ARGS.out, best["params"])
        print(f"[tune] wrote {ARGS.out}  (apply with -Xptxas=--apply-controls={ARGS.out})")
    else:
        print("[tune] best is not better than the default schedule; nothing written")


if __name__ == "__main__":
    main()
