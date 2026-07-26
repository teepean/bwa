# CUDA port — progress log

Goal: massively speed up `bwa aln` (BWA-backtrack) for ancient DNA, producing a **bit-exact**
`.sai` vs CPU bwa. Single-end only. Target GPU: RTX 3090 (sm_86, 24 GB). Gold-standard command:
`bwa aln -l 1024 -n 0.01 -o 2`. See `../CUDA_PORT_PLAN.md` for the full research + architecture.

## Test assets (not committed; under `../test_data/`, regenerate as needed)
- Reads: `/home/dnastorage/aDNApipeline/AVA1B/AVA1B.combined.fq.gz` (merged aDNA, 30–91 bp, mean 46).
- Index: `/home/dnastorage/aDNApipeline/hs37d5.fa` (.bwt 3.14 GB; .sa NOT needed on GPU).
- Subsets: `sub10k.fq`, `sub100k.fq` (deterministic heads), `strided20k.fq` (representative).
- Golden CPU `.sai` for sub10k: md5 `54410c755509b8389ae992953dd3476c`.

## Phase 0 — baseline & profiling — DONE
Instrumented `bwt_match_gap` (`#ifdef ALN_PROFILE` in `bwtgap.c`; build `bwa-prof`). Verified
bit-exact `.sai` vs clean build. Findings (representative 20k sample):
- Mapping rate **0.52%** → **99.3% of reads have zero hits** and dominate runtime.
- **No read hit the 2M `max_entries` cap** (worst peak queue 1.08 M, worst 1.27 M expansions; mean ~40 k).
- Tree size spans **6 orders of magnitude**.
- ⇒ For zero-hit reads (cap never reached), DFS and best-first visit the IDENTICAL node set; the
  priority queue is unnecessary for the bulk. Architecture: persistent-thread **work-pool** + per-read
  iterative DFS (small local stack); flag the ~0.5% reads that find a hit and reconcile via the exact
  CPU best-first path for bit-exactness. Load balancing is the #1 problem.

## Phase 1 — device FM-index (Occ) — DONE
`cuda/fm_device.cuh`: device mirrors of `bwt_occ4` / `bwt_2occ4` / `bwt_occ` / `bwt_match_exact`,
operating on the existing 64-byte-bucket layout; `cnt_table`/`L2` in constant memory; `.bwt` uploaded
to global memory (3.14 GB); reads via `__ldg`. `cuda/fmtest.cu` validates against CPU.
Build: `make fmtest`  ·  Run: `./fmtest <ref.fa> [reads.fq] [n_random_occ]`.

Result on hs37d5 + sub10k:
- Test A (random Occ4, 10 M probes): **PASS, 0 mismatches, 2.28 G-Occ4/s**.
- Test B (full backward exact-search, 10 k real reads): **PASS, 0 mismatches** (42 full-length hits).

## Phase 2 step 1 — bit-exact device DFS (hit detection) — DONE (correctness), perf baseline only
`cuda/dfstest.cu` + `cuda/fm_device.cuh` (added `d_bwt_match_exact_alt`, `d_int_log2`). Host reuses
bwa's real I/O + `bwt_cal_width` + complement + per-read `max_diff`, then the device DFS
(`d_dfs_has_hit`) reproduces `bwt_match_gap`'s bounded node set and reports has_hit; the ~0.5% hit
reads are reconciled by the exact CPU `bwt_match_gap` for a bit-exact `.sai`.
Build `make dfstest`; run `./dfstest <ref.fa> <reads.fq> [golden.sai]` (replicates `-l 1024 -n 0.01 -o 2 -t 16`).

Result on sub10k vs golden (md5 `54410c75…`):
- check1 CPU-only `.sai` == golden ✓ (harness == driver)
- check3 GPU has_hit=52, **false_neg=0**, false_pos=0 ✓ (DFS detection complete & exact)
- check2 **hybrid `.sai` == golden** ✓ — **bit-exact achieved**
- stack_overflow=0 (cap 8192 entries sufficient).

**Performance baseline (naive, one-read-per-thread grid-stride, P=8192, global-memory stack):**
**100 reads/s** — vs ~3,300 reads/s on the 16-core CPU → **~33x SLOWER**. This reproduces the known
BarraCUDA-style failure mode. Diagnosed causes for step 2:
1. **Warp divergence**: 32 reads per warp with 6-orders-of-magnitude tree-size variance → warp waits
   for its slowest read; one 1.27M-node read stalls 31 idle lanes.
2. **Low occupancy / poor latency hiding**: only 8192 threads; FM-index probes are ~500 ns random
   global reads that need far more in-flight warps to hide.
3. **Pathological stack memory pattern**: per-thread stacks at `slot*cap` stride (196 KB apart) →
   every push/pop is 32 uncoalesced cache lines; ~5-9 stack writes per node dominate traffic.
4. `d_bwt_2occ4` = 2 full occ4 (no shared-bucket fast path).

Reference ceiling: ~1e9 occ4 for the whole 10k set; at the measured 2.28 G-Occ4/s that's ~0.44 s if
perfectly parallel → the naive run is ~227x off the FM-index ceiling, i.e. essentially all loss is
scheduling/memory, not the index. Headroom is enormous.

## Phase 2 step 2 — performance — IN PROGRESS (bit-exact maintained throughout)

Changes to `cuda/dfstest.cu`: coalesced SoA stack (`[sp*P+slot]`), persistent-thread work-pool
(atomic work counter), occupancy-sized launch (82 SM x 7 blk x 128 = 73472 slots), per-read **work
budget** (default 2M pops via `DFS_BUDGET`) that flags pathological reads to the exact CPU path,
and an opt-in full CPU reference (`DFS_FULLCPU=1`, auto-on for <=20k reads) — large runs reconcile
only the flagged/hit reads from stored flat data (the real production hybrid).

Findings (the key one): the naive 100s was **dominated by ONE read** doing 33.1M node-pops on a
single thread. The CPU's `max_entries=2M` frontier cap makes `bwt_match_gap` bail on such reads
(returning n_aln=0), so CPU stays fast; the GPU had no equivalent. Coalesced stack + occupancy +
work-pool ALONE changed nothing (still ~100s) because time was that one serial read, not
scheduling. The **work budget** fixed it — and it's free for bit-exactness because flagged reads
are reconciled by the exact CPU search anyway.

Measured (RTX 3090, hs37d5, `-l 1024 -n 0.01 -o 2`, budget 2M):
| set | naive | +budget | flagged | bit-exact |
|-----|-------|---------|---------|-----------|
| sub2k  | 101.8 s (20 r/s) | 4.5 s (440 r/s) | 2 (0.1%) | yes |
| sub100k | — | **12.08 s (8276 r/s)** + 2.37 s reconcile (520 reads, 1 thread) | 11 budget + ~509 hits | **yes (md5 eecf35c1)** |

CPU baseline on sub100k, same box: **`bwa aln -t 16` = 23.47 s (4261 r/s)**.
=> GPU kernel is **~1.9x the full 16-core CPU** already (reconcile parallelizes/overlaps away);
**~83x** over the naive GPU baseline. Throughput 338M pops/s ~= 38% of the FM-index occ4 ceiling,
so meaningful headroom remains.

### step-2 optimization log (sub100k, RTX 3090, bit-exact throughout)
| change | reads/s | note |
|--------|---------|------|
| work-budget engine (above) | 8276 | baseline for this log |
| + `bwt_2occ4` shared-bucket fast path | 8224 | **no change** -> not occ-load bound |
| maxrregcount 48 (40 warps) | 6445 | **slower** |
| maxrregcount 40 (48 warps) | 5243 | **slower** -> NOT occupancy/latency bound |
| + in-register-continue DFS (stack only for siblings, pop only on backtrack) | **9934** | +20%; 69 regs |

Diagnosis (consulted CUDA Ampere tuning guide + NVIDIA forums + Volkov latency-hiding + the 2025
arXiv "N-Queens GPU iterative DFS" paper): the kernel is **bound by the global-memory DFS stack**,
not occ loads and not occupancy. Evidence: the 2occ4 fast path didn't help, and *raising* occupancy
*hurt* (more concurrent threads -> the per-thread live stack working set exceeds the 6 MB L2 ->
DRAM). The literature is explicit: one-thread-per-task DFS with a global stack scales poorly;
assign a subtree to a group of threads sharing fast memory, and keep the stack in shared/registers.
The in-register-continue change applies the register part (descend in registers, push only siblings,
read stack only on backtrack) for +20%.

Current standing: **9934 reads/s = ~2.3x the 16-core CPU (4261 r/s), ~27x one core (363 r/s),
99x over the naive GPU baseline (100 r/s); bit-exact.** Throughput ~408M pops/s ~= 45% of the
occ4 ceiling.

Next big levers (toward "massive"): (1) **one-read-per-warp / subtree-per-warp cooperation** with the
hot stack in shared memory (the literature's recommended structure; should cut both divergence and
the L2-thrashing stack); (2) shared-memory hot-stack window backing the global spill; (3) full-file
streaming with overlapped, multithreaded CPU reconcile. References saved in cuda/REFERENCES.md.

### step-2 cont. (data that pins down the next move)
- **DFS stack DEPTH (sub100k): mean 111, max 386.** Histogram concentrated at <=128 (90,876 reads),
  <=256 (7,612), <=512 (223). So a **512-entry stack covers every read** -> a per-warp shared-memory
  stack is feasible (~512*20 B = 10 KB/warp). Reduced cap 1024->512 (0 overflow).
- **Block-local stack re-striding ([blockIdx*blockDim*cap + d*blockDim + tib]): NO win** (9165 vs
  9934 r/s, bit-exact). Negative result -> the global stack is bound by traffic VOLUME/bandwidth,
  not layout/locality. Re-striding can't reduce volume; only removing global stack traffic can.
- A per-*thread* full shared stack can't fit (128 thr * 386 * 20 B ~= 1 MB/block). So the shared
  stack **requires one read per warp** (one stack/warp) -> which forces lane cooperation = Idea #1.
  Decision: implement the warp-cooperative engine with a per-warp shared-memory stack.

## Phase 2 step 2 — WARP-COOPERATIVE engine (Idea #1+#2) — DONE (kernel), bit-exact
`DFS_ENGINE=warp` in `cuda/dfstest.cu`: one read per warp; 32 lanes co-explore the tree as a
frontier; per-warp stack in SHARED memory (SoA). Wave = pop up to 32 top nodes -> expand in
parallel -> warp prefix-sum compact children -> push to shared stack; hit via `__any_sync`;
budget/overflow -> CPU reconcile. Removes global stack traffic.

**sub100k GPU kernel: 29,708 reads/s** (CAP=640, 4 warps/blk) = **3x the thread engine (9934),
~7x the 16-core CPU (4261 r/s)**; bit-exact (md5 eecf35c1). sub2k: 6304 vs 547 r/s (11.5x) —
monster reads now spread across 32 lanes instead of serializing on one thread.

### CURRENT BOTTLENECK (documented) — CPU reconcile of overflow-flagged reads
The warp wave pops 32 nodes and pushes ALL their children, so the live frontier is **BFS-wide**,
not bounded by the serial DFS depth (386). With the shared stack CAP=640 it **overflows for 4.6%
of reads**, which are flagged to the CPU. Reconciling those 4,835 bushy reads on **one CPU thread
takes 84 s** — it now *dwarfs* the 3.4 s GPU kernel (end-to-end 3.4 + 84 = 87 s, worse than CPU's
23.5 s). So the bottleneck has MOVED from "GPU global-stack traffic" to "CPU reconcile of
overflow-flagged reads"; the GPU itself is no longer the limiter.

Levers (both needed):
1. **Cut the overflow flag rate** toward the real-hit floor (~0.5%): raise CAP (bounded by shared
   mem -> occupancy), and/or bound the warp frontier so it stays near the serial DFS depth (e.g.
   pop fewer per wave / drain depth-first when near capacity) so CAP=512 suffices.
2. **Multithread + GPU-overlap the reconcile** (Idea #5): the flagged-read CPU work is embarrassingly
   parallel; 16 threads -> ~5x, and it can overlap the next GPU batch via streams. Even at 4.6%,
   84 s/16 ~= 5.3 s; with flag rate driven to ~0.5% it becomes negligible and the engine is
   GPU-bound at ~29.7k r/s (~7x CPU).
Added `DFS_NORECON` to skip reconcile during perf sweeps.

### Resolution of the bottleneck (CAP tuning + multithreaded reconcile)
CAP sweep (sub100k, warp, 4 warps/blk): CAP=640 -> 4.6% flagged; CAP=1024 (80 KB) -> 0.28%;
CAP=1216 (95 KB) -> 0.065%. Larger CAP = GPU does the bushy reads itself (fewer flags) but is a
bit slower (more GPU work); the CAP=640 "34k r/s" was illusory (it punted bushy reads to a 84 s
CPU reconcile). Default CAP set to 1024.
**Multithreaded the reconcile** (std::thread, per-thread gap_stack; bwt_match_gap is thread-safe):
751 flagged reads reconciled in **1.57 s wall on 16 threads** (was 12.1 s single-thread).

**Engine standing (sub100k, RTX 3090, bit-exact md5 eecf35c1):**
- Warp GPU kernel: **21,432 reads/s** (CAP=1024).
- End-to-end GPU + 16-thread reconcile (sequential): 4.67 + 1.57 = 6.24 s = **16,026 reads/s**.
- With streaming overlap (reconcile hidden behind next GPU batch): GPU-bound ~**21,432 r/s**.
- CPU `bwa aln -t 16` = 4,261 r/s on the same box -> **~3.8x (sequential) to ~5x (overlapped)**.
- vs the naive GPU baseline (100 r/s): ~210x.

Remaining levers: (a) shared(small)+global-backing two-level stack -> high occupancy (currently
only 4 warps/SM due to the 80 KB shared stack) without flagging, to push the GPU beyond 21k;
(b) full-file streaming engine with double-buffered batches + reconcile overlapping the next
kernel (to realize the GPU-bound rate and process the whole 3.95M-read file end-to-end).

### Drain-down + carry register-continue (Lever 1) + the end-to-end ceiling
Implemented dynamic pop-count: WAVE mode pops up to 32 nodes but only while room=(CAP-sp)/9 allows
every node's <=9 children (so wave mode NEVER overflows); DRAIN mode (near full) pops 1 and keeps a
child in registers (carry) so single-chains never touch the stack. Gated carry to sp>=CAP/2 to keep
the wave ramp-up for normal reads. MAX_CHILDREN confirmed = 9 (state M: 1 ins + 4 del + 4 mm).
All bit-exact (sub2k false-pos 90->1).

CAP sweep (carry-drain, sub100k, GPU-only): CAP=256 -> 27% flag, 59k r/s (16 warps/SM);
384 -> 5.6%, 35.7k (12 w/SM); 512 -> 2.2%, 21k (8 w/SM); 768 -> 0.34%, 22.2k (4 w/SM).
With the overlapped 16-thread reconcile (~444 flagged-reads/s), end-to-end is reconcile-bound for
small CAP and GPU-bound (~22k) for CAP>=768. **KEY: the high-occupancy small-CAP GPU speed is
illusory — it punts the bushy reads (2-5%) to the CPU. Those reads cost similarly on GPU (low
occupancy) or CPU (reconcile), pinning end-to-end at ~22k r/s (~5x the 16-core CPU) regardless of
CAP.** Default CAP set to 768 (0.34% flag, GPU-bound 22.2k).

To break the ~22k ceiling, the bushy reads must run ON THE GPU AT HIGH OCCUPANCY -> the two-level
**shared(small, e.g. 256 -> 16 warps/SM) + per-warp global-backing** stack: 95%+ of reads stay in
shared (59k base rate); the 2-5% bushy reads spill their deep frontier to global (handled on GPU,
NOT flagged). Only true 2M-budget reads -> CPU. This is the next build.

## Phase 2 step 2 — TWO-LEVEL stack engine (`DFS_ENGINE=warp2`) — DONE, bit-exact
Per-warp shared top-window (CAP_SM) + pre-allocated per-warp GLOBAL backing; warp-parallel
coalesced spill/unspill of 128-entry chunks (invariant: global=oldest bottom, shared=newest top).
Bushy reads spill to global and STAY ON THE GPU instead of flagging to CPU.

Result (sub100k, bit-exact md5 eecf35c1): **flag rate driven to ~0.015%** (vs 0.34% single-level)
-> CPU reconcile negligible (520 reads, 0.61 s). CAP_SM sweep: 256->15.9k r/s (16 w/SM),
384->18.8k (12), 512->21.5k (8), 768->22.1k (4).

**KEY FINDING — we are at the FM-index occ4 ceiling, NOT occupancy-bound.**
> **[SUPERSEDED 2026-07-26 — see Phase 6 / D6.]** The "occ4 ceiling" quoted here (2.28-2.32 G/s)
> was measured at HIGH occupancy. At this kernel's actual 8 warps/SM the hardware delivers
> 5.84 G probes/s. The kernel was never at the ceiling; the real limiter was 320 B/thread of
> local memory (D5). The occupancy-insensitivity observed below is real, but its cause was
> local-memory/L1 contention plus spill traffic, not an FM-index bandwidth ceiling.

Occupancy from 4 to 16
warps/SM does NOT raise throughput; all engines/configs plateau at **~22k reads/s**. This work is
~8e9 occ4 (4.1e9 node-pops x ~2 probes); at the measured 2.28 G-occ4/s that is a ~3.6 s floor vs
~4.5 s actual (~80% of ceiling, ~5x the 16-core CPU). The 40-50k target is not reachable on this
GPU because the kernel is bound by FM-index probe throughput, not latency hiding. Raw speedup
beyond this needs FEWER occ4 per node (algorithmic; constrained by bit-exactness) or a faster
FM-index (k-step / occ caching) -- diminishing returns. Default CAP_SM=512.

**warp2 is the engine to ship**: same ceiling speed as warp1 but ~0 CPU reconcile tail, which keeps
the streaming engine clean. Next: full-file streaming (#2) -- the deliverable, not a speed lever.

## Phase 2 step 2 — STREAMING full-file tool `bwa-aln-gpu` — DONE, bit-exact end-to-end
`cuda/aln_gpu.cu` + `cuda/dfs_engine.cuh` (extracted warp2). Streams the FASTQ in bwa's native
0x40000-read chunks; per chunk: MT preprocess (bwt_cal_width + complement + per-read max_diff) ->
warp2 GPU has_hit -> MT CPU reconcile of flagged/hit reads -> write records in read order.
`make bwa-aln-gpu`; `./bwa-aln-gpu [-l -n -o -t -f] <ref.fa> <in.fq>`.

**FULL FILE (AVA1B.combined.fq.gz, 3,948,528 reads, hs37d5, -l 1024 -n 0.01 -o 2, RTX 3090):**
- bwa-aln-gpu: **175.9 s = 22,445 reads/s** (preprocess 1.4s, gpu 156.5s, reconcile 16.7s/16thr, io 0.1s);
  0.531% reconciled on CPU. Output .sai = 16.85 MB.
- CPU `bwa aln -t 16` golden on the same file: **875 s**.
- => **4.97x end-to-end speedup, BIT-EXACT**: both .sai = 16,852,676 bytes, md5
  `a09a26dd894690031da42a00862f7a3d` (GPU == CPU). Full 3.95M-read file verified byte-identical.

## Phase 3 — FUSED `alnse` (aln+samse, multithreaded) — `bwa-aln-gpu -S`
Added `-S` (SAM out) + `-r RG` to `cuda/aln_gpu.cu`: fuses aln and samse in one command
(GeoGenetics-style alnse). No `.sai` on disk; the FASTQ is read ONCE (samse reuses the loaded
`seqs[]`); `bns`/SA/`pac` loaded once. Per chunk after GPU aln + reconcile:
- serial `bwa_aln2seq_core` (preserves `drand48` repeat-hit-selection order; the only samse RNG --
  `lrand48` at bntseq.c:266 is index-build only),
- **MT** `bwa_cal_pac_pos_core` SA-lookup (RNG-free) across `n_threads`,
- serial `bwa_refine_gapped` + `bwa_print_sam1` (ordered, bit-exact output).
Header via `bwa_print_sam_hdr` (@HD/@SQ/@RG) + a `@PG` for bwa-aln-gpu.

Validated: fused SAM **alignment records are byte-identical** to `bwa samse` on the bit-exact `.sai`
(sub100k md5 `ee2dd6ef…`; header differs only in the expected @PG CL). `... -S | samtools sort` -> BAM.

**FULL FILE fused alnse (3,948,528 reads, RTX 3090): 174.2 s = 22,672 reads/s**, SAM records
BIT-EXACT vs reference samse (md5 `e2a6e1c9…`). preprocess 1.4s, gpu 153.9s, reconcile 16.8s, io 0.8s.
The fused run (174 s) is FASTER than aln-only(176s)+separate samse(20s)=196s -- samse folds in for
free (single FASTQ read, MT SA-lookup, samse compute hidden). vs CPU `bwa aln -t16`+samse = 895 s
=> **~5.1x end-to-end, one command, no intermediate .sai, byte-identical alignments.**
Usage: `bwa-aln-gpu -S -r '@RG\t...' ref.fa reads.fq.gz | samtools sort -O bam -o out.bam -`

## Phase 3b — CPU/GPU overlap (#5) — DONE, bit-exact
`cuda/aln_gpu.cu` restructured into a producer/consumer pipeline: the main thread owns the GPU
(read -> MT preprocess -> upload -> kernel -> download has_hit, serial on one stream so the shared
global backing is never raced); each chunk is handed to ONE in-order finisher thread that does the
CPU reconcile + output (.sai or samse), running concurrently with the next chunk's GPU work. Single
ordered consumer preserves drand48/output order -> bit-exact. (Multi-GPU-ready: replicate the GPU
producer stage per device + round-robin chunks; keep one ordered finisher.)

Full file (3.95M reads, fused alnse SAM): **174.2 s -> 160.9 s = 24,541 reads/s** (overlap hid ~13s
of the ~18s CPU tail), records BIT-EXACT (md5 e2a6e1c9). Both modes re-verified bit-exact under the
pipeline (sub100k .sai eecf35c1, SAM records identical). vs CPU `bwa aln -t16`+samse = 895 s ->
**~5.56x end-to-end.**

## Phase 3c — native `bwa gpualn` subcommand — DONE
`cuda/aln_gpu.cu`'s entry is now `extern "C" int bwa_alnse_gpu(int,char**)`; `main.c` dispatches
`bwa gpualn` under `#ifdef HAVE_CUDA`. Builds:
- `make`           -> CPU-only `bwa` (NO CUDA dependency; `gpualn` absent) -- fork still builds anywhere.
- `make bwa-gpu`   -> CUDA `bwa` with the `gpualn` subcommand (main.c -DHAVE_CUDA + nvcc aln_gpu.o,
                      linked via nvcc; reuses the existing AOBJS for samse/aln CPU functions).
- `make bwa-aln-gpu` -> standalone tool (unchanged; -DALN_GPU_MAIN).
Usage:
  bwa gpualn [-l 1024 -n 0.01 -o 2 -t 16] ref.fa reads.fq.gz > out.sai           # .sai (like bwa aln)
  bwa gpualn -S -r '@RG\t...' ref.fa reads.fq.gz | samtools sort -O bam -o out.bam -   # fused alnse -> BAM
Verified: `bwa gpualn` .sai md5 == golden (eecf35c1); `bwa gpualn -S` SAM records == `bwa samse`;
@PG is proper bwa provenance (ID:bwa). CPU-only `bwa` regression-checked (no gpualn, no CUDA link).

## Validation vs the PRODUCTION BAM (/home/dnastorage/aDNApipeline/AVA1B_aln/)
Reference: `AVA1B_aln.short.bam` = production `bwa aln -l 1024 -n 0.01 -o 2` + `samse` + sort, built
with bwa **0.7.18-r1243-dirty** on the same 3,948,528 reads (20,701 mapped, 0.52%). Compared to
`bwa gpualn -S` output (this tree): extracted mapped records, dropped the RG:Z tag (production used
a different @RG ID), sorted by QNAME:
- mapped count identical (20,701 vs 20,701); mapped read SET identical (same QNAMEs) -> no read
  changed mapped/unmapped status.
- **ALL 20,701 mapped records + ALL tags (FLAG/RNAME/POS/MAPQ/CIGAR, XT/NM/X0/X1/XM/XO/XG/MD)
  byte-identical.** Holds across the 0.7.18-dirty -> 0.7.19 gap (aln/bwtgap core unchanged).
Only the @RG ID and the SAM @PG line differ -- nothing in the alignments. End-to-end equivalence to
the production pipeline confirmed.

## STATUS: GOAL ACHIEVED
GPU `bwa aln` (BWA-backtrack) for ancient DNA at `-l 1024 -n 0.01 -o 2`, single-end:
**~5x the 16-core CPU on a full real file, byte-identical .sai, GPU-bound at the FM-index ceiling.**
Engine: warp-cooperative two-level-stack DFS (`cuda/dfs_engine.cuh`) + MT CPU reconcile of ~0.5%;
tool: `bwa-aln-gpu` (`cuda/aln_gpu.cu`). The earlier BarraCUDA failure mode (slow with seeding off)
is resolved: this is fastest precisely in the seeding-off aDNA regime.

Possible future work (optional, diminishing/oncosting returns): double-buffered stream overlap of
preprocess/reconcile with the kernel (~10% — preprocess+reconcile is only ~18s of 176s); faster
FM-index (k-step / L2-pinned occ) to lift the ~22k occ4 ceiling; multi-GPU; wrap as a real
`bwa aln` subcommand/flag in main.c; libdeflate FASTQ decode (not currently a bottleneck, ~1.2s).

## Phase 4 — PIGEONHOLE PREFILTER (break the occ4 ceiling by eliminating work) — IN PROGRESS

### Insight
The ceiling analysis (HBM random-access bandwidth, ~2.28 G occ4/s) is correct *given* that every
read runs the full DFS. But 99.3% of reads are unmappable, and we spend ~40k node-pops (~80k occ4)
per read *proving* that. A cheap necessary-condition check can eliminate them BEFORE the DFS.

### Pigeonhole principle (lossless filter)
For `-n 0.01` and reads 30–91 bp: `max_diff = int(L*0.01 + 0.999) = 1` for every read.
A read that maps with ≤ max_diff error events (mismatch or gap open) must contain at least one
error-free contiguous segment of length ≥ floor(L / (max_diff+1)).

Proof: max_diff errors can disrupt at most max_diff of the (max_diff+1) contiguous segments
(pigeonhole); at least one segment is untouched and matches the reference exactly. This holds for
gaps too: a gap open at position p disrupts only the segment containing p; segments entirely on
one side still match a contiguous reference substring (at a shifted position).

Therefore: if NO floor(L/(max_diff+1))-mer of the read exists in the reference (exact FM-index
backward search), the read CANNOT map within max_diff errors. Zero false negatives → bit-exact.

### Cost analysis
- Check (max_diff+1) = 2 non-overlapping segments of length k = floor(L/2).
- Each exact backward search terminates in ~12–15 bases when the interval goes empty.
- **~30 occ4 probes per filtered read vs ~80k for the full DFS → ~2600x fewer probes per read.**

### Filtering power (for unmappable reads)
P(random k-mer exists in 3.1 Gbp ref) ≈ seq_len / 4^k.
- k=23 (L=46): ≈ 4.4×10⁻⁷. With 2 segments: P(passes) ≈ 9×10⁻⁷.
- k=45 (L=91): ≈ 2.5×10⁻¹⁸. Essentially 0.
→ **99.9999%+ of unmappable reads are eliminated by the prefilter.**

### Projected impact (sub100k)
| | occ4 total | time @ 2.28 G/s | reads/s |
|---|---|---|---|
| Current (all DFS) | ~8×10⁹ | 4.5 s | 22k |
| Prefilter (99.3k×30) + DFS (700×80k) | ~59M | ~26 ms | ~3.8M (theoretical) |

Realistic (overhead, serialization of hit reads): **100k–1M+ reads/s** (50–100x over current).

### Implementation
- In `d_dfs_has_hit_warp2`: before the DFS loop, lanes 0..max_diff each run an exact backward
  search on one segment. `__any_sync` — if ANY segment matches → proceed to DFS. If NONE → return 0.
- Toggle: `DFS_NOPREFILTER=1` disables for A/B comparison.
- Instrumentation: `n_prefiltered` counter in the kernel.

### Correctness argument
The filter is a NECESSARY condition for mappability (pigeonhole). It can only produce:
- True negatives (read is unmappable AND filter says so) → skip DFS, has_hit=0. CORRECT.
- False positives (read is unmappable BUT filter passes) → DFS runs, finds no hit. CORRECT.
- True positives (read is mappable AND filter passes) → DFS runs, finds hit. CORRECT.
- False negatives: IMPOSSIBLE by the pigeonhole proof above.
→ Bit-exactness is preserved regardless of filter behavior.

### NEGATIVE RESULT — prefilter is ineffective for this parameter regime
The analysis above assumed max_diff=1 (from the naive formula `int(L*fnr+0.999)`). But bwa's actual
`bwa_cal_maxdiff` uses a Poisson model: `bwa_cal_maxdiff(L, BWA_AVG_ERR=0.02, fnr=0.01)`, which gives:

| L | max_diff | nseg=L/(md+1) | seg_len | P(seg exists in 3.1Gbp ref) |
|---|----------|---------------|---------|----------------------------|
| 30 | 3 | 4 | 7 | 1.0 (4^7=16K << 3.1G) |
| 46 | 4 | 5 | 9 | 1.0 (4^9=262K << 3.1G) |
| 60 | 4 | 5 | 12 | 1.0 (4^12=16.7M << 3.1G) |
| 91 | 6 | 7 | 13 | 1.0 (4^13=67M << 3.1G) |

Every pigeonhole segment (7–13 bp) is guaranteed to exist somewhere in a 3.1 Gbp reference.
The filter passes 100% of reads → zero filtering power. Confirmed empirically: sub2k run shows
"pigeonhole-prefiltered: 0 / 2000 (0.00%)".

**Root cause:** the pigeonhole filter requires segments of length floor(L/(max_diff+1)) to be
ABSENT from the reference to prove unmappability. This only works when 4^seg_len >> seq_len,
i.e., seg_len > log4(3.1e9) ≈ 16. For max_diff=3–6 and L=30–91, seg_len=7–13 < 16. The filter
is structurally unable to eliminate any read.

**The pigeonhole prefilter would only be effective for:** max_diff ≤ 1 (e.g., `-n 0` exact mode,
or very short reads <20bp) or much larger genomes where seg_len > 16. NOT for aDNA at -n 0.01.

Code is left in place (gated by `DFS_NOPREFILTER`, default ON but filtering 0%) as it costs
negligible overhead (~2 occ4 probes/read for the two exact searches that always succeed) and
would activate for other parameter regimes. The instrumentation counter confirms the 0% rate.

### Revised assessment of remaining optimization paths
With the prefilter ruled out, the system remains at the HBM random-access bandwidth ceiling
(~2.28 G occ4/s, ~22k reads/s). The 40k node-pops per unmappable read (max_diff=3–6) is
irreducible: proving a read unmappable requires exhausting the full bounded DFS tree.

Remaining levers (all hardware/representation, not algorithmic):
1. **Multi-GPU** (~2× per additional GPU; doubles HBM bandwidth). Embarrassingly parallel.
2. **2-bit compressed BWT** (~1.5–2×): pack BWT symbols 2-bit (vs current byte-in-bucket),
   reducing the 64-byte bucket to ~48 bytes → fewer cache lines per probe → higher effective
   bandwidth. Requires re-packing at load time (one-time cost). The occ computation changes
   (bit extraction instead of cnt_table lookup) but is compute-cheap.
3. **L2 persistence for root buckets** (~5–10%): pin the first-level SA interval buckets via
   `cudaAccessPolicyWindow`. Every backward search's first probe hits one of 4 buckets.
4. **k-step FM-index** (marginal, risks bit-exactness): precompute 2-base transitions to halve
   probe count. Table size prohibitive for full genome; only feasible for sampled positions.

### Vectorized checkpoint loads (uint4) — NEGATIVE RESULT
Replaced the four scalar `__ldg(pc+i)` u64 loads with two 128-bit `uint4` loads + shift/OR
reconstruction in both `d_bwt_occ4` and `d_bwt_2occ4` (shared-bucket path). Bit-exact (sub2k
md5 4b068014). sub100k: **20,847 reads/s** vs documented 21,432 baseline → **no change** (within
noise). The shift+OR reconstruction adds instructions that offset any load-throughput gain.
Reverted. Confirms the kernel is NOT instruction/issue-bound — it is purely HBM-bandwidth-bound.

### L2 persistence for root buckets — NOT IMPLEMENTED (analysis shows it cannot help)
The "root" BWT buckets (at c_L2[c] positions, accessed by every read's first 1–2 probes) are
scattered across the full 3.14 GB array (NOT contiguous). `cudaAccessPolicyWindow` requires a
contiguous range, so it cannot pin these 4–5 scattered buckets. Moreover, as already documented:
"cross-read hot-root blocks are <0.1% of probes and already L2-resident" — the hardware LRU
naturally keeps them cached (every warp hits them). Pinning would save <0.1% of probes. Not worth
the implementation complexity.

### FINAL CONCLUSION — single-RTX-3090 ceiling is hardware-bound
> **[SUPERSEDED 2026-07-26 — see Phase 6 / D5-D7.]** This conclusion is WRONG. It rests on the
> high-occupancy 2.28 G-occ4/s figure; the true rate at the shipped occupancy is 5.84 G probes/s
> (D6). A pure code-generation fix (removing 320 B/thread of address-taken local memory, plus
> popc) gave **1.82x with no algorithmic or hardware change**, and ~2.4x of non-bandwidth
> headroom still remains. The three attempts cited below (pigeonhole prefilter, vectorized
> loads, L2 persistence) each failed for their own reasons -- none of them established a
> bandwidth wall.
Three independent optimization attempts (pigeonhole prefilter, vectorized loads, L2 persistence)
all confirmed: the kernel is at the HBM random-access bandwidth wall. The 64-byte bucket is
already the minimum cache-line-granularity read (u64 checkpoints required since seq_len≈6.28G >
uint32_max; BWT already 2-bit packed; bucket = exactly one 64-byte line). The 40k node-pops per
unmappable read is algorithmically irreducible at max_diff=3–6. No single-GPU software change can
break the ~22k reads/s ceiling.

**The only paths beyond 22k reads/s are:**
1. **Multi-GPU** (linear scaling; the streaming engine is already chunk-parallel-ready).
2. **A fundamentally different FM-index representation** that reduces bytes-per-probe below 64
   (would require a custom index format, not bwa-compatible; marginal gain since 64B = 1 sector).
3. **A different GPU with more HBM bandwidth** (e.g., A100 80GB at 2 TB/s → ~2.4× over 3090).

## Phase 5 — MULTI-GPU pipeline — DONE (implementation), pending 2-GPU benchmark

### Hardware
- GPU 0: RTX 3090 (24 GB, 936 GB/s HBM, 82 SM)
- GPU 1: RTX A4000 (16 GB, 448 GB/s GDDR6, 48 SM) — pending install
- Motherboard: 2× PCIe 4.0 x16

### Architecture (3-stage pipeline in `cuda/aln_gpu.cu`)
- **Stage 1 (main thread):** read FASTQ chunk + MT preprocess (bwt_cal_width + complement) →
  push `Chunk*` to bounded "ready" queue (capacity = nGpu+2).
- **Stage 2 (N GPU worker threads, one per device):** pop from ready → `cudaSetDevice(g)` →
  upload seq/width/readparam → launch `k_dfs_warp2` → sync → download `has_hit` → mark chunk
  done in ordered completion buffer (`cslots[seq_id]`).
- **Stage 3 (single ordered finisher thread):** wait for `cslots[next_out]` → CPU reconcile
  (MT `bwt_match_gap`) → output (.sai or samse) → advance `next_out`.

Bit-exactness: the finisher processes chunks strictly in read order (sequence-numbered), so
`drand48` RNG order and output order are identical to the single-GPU path. Each GPU has its own
BWT copy (3.14 GB fits in both 24 GB and 16 GB), its own global backing, and its own device
buffers. Work distribution is dynamic (shared ready queue): the faster 3090 naturally pulls more
chunks.

### Expected throughput
- 3090: ~22k reads/s (measured)
- A4000: ~22k × (448/936) ≈ ~10.5k reads/s (HBM-bandwidth-proportional)
- Combined: ~32.5k reads/s ≈ **1.48× single 3090, ~7.6× the 16-core CPU**

### Usage
Auto-detects all visible GPUs. Override with `GPUALN_NGPU=N` to limit.
```
./bwa-aln-gpu -t 16 ref.fa reads.fq.gz > out.sai          # all GPUs
GPUALN_NGPU=1 ./bwa-aln-gpu ...                           # force single GPU
```

### Validation (single-GPU mode, refactored pipeline)
- sub2k: md5 `4b068014` == golden ✓
- sub100k: md5 `eecf35c1` == golden ✓ (16,906 reads/s; slightly lower than old single-thread
  GPU path due to the queue overhead, but the multi-GPU overlap will more than compensate)

### Pending
- Install A4000, run full 3.95M-read file with both GPUs, verify bit-exact + measure speedup.

## Phase 6 — ROUND-3 DIAGNOSTICS (measurement first) — the "occ4 ceiling" conclusion was WRONG

Round 2 concluded the kernel sits at the HBM random-access wall and that only multi-GPU /
faster hardware could help. Direct measurement refutes this: **the kernel runs at 39-50% of the
FM-index probe ceiling**, and the missing half is *local memory*, not DRAM bandwidth.

### Tooling added
- `make bwa-aln-gpu-instr` — instrumented twin (`-DDFS_INSTRUMENT`). Production codegen is
  untouched (all instrumentation is `#ifdef`-guarded, so register count is unchanged).
- Restored the per-length-band histogram in `cuda/aln_gpu.cu`: the `H_*` vectors were declared
  but never filled after the Phase-5 pipeline refactor (`d_npop`/`d_flag` were never copied
  back), so `GPUALN_HISTO=1` reported `reads=0`. Now accumulated in the finisher.
- New counters: FM probes, 64 B buckets touched, node pops, expand waves, spill events, and a
  sampled (depth, errors-used) histogram of node pops.

### D1 — working-set sweep (`FMTEST_KRANGE`): NOT a TLB / locality effect
| BWT touched | 3.14 GB | 500 MB | 250 MB | 100 MB | 25 MB | 5 MB | 1 MB |
|---|---|---|---|---|---|---|---|
| G-occ4/s | 2.32 | 2.35 | 2.39 | 2.17 | 3.24 | 8.28 | 10.17 |

Throughput is **flat from 3.14 GB down to a 25 MB working set** and only lifts when it fits in
L2 (6 MB). A 25 MB footprint is ~13 huge pages, so address translation is not the limiter.
Shrinking the index buys nothing; only bytes-per-probe does.

### D2 — random-gather microbenchmark (3 GB buffer, RTX 3090)
| probe size | throughput | useful bandwidth |
|---|---|---|
| 64 B (bwa's bucket) | 2,243 M/s | 143.5 GB/s |
| 32 B | **4,230 M/s** | 135.4 GB/s |

Halving bytes-per-probe gives **1.89x more probes/s**. Also: `fmtest`'s `d_bwt_occ4` achieves
2.32 G/s, i.e. *at* the raw 64 B gather rate — so the Occ code itself has no recoverable
overhead, and the divergent `c_cnt_table` constant-memory lookups are already fully hidden.
**=> replacing `cnt_table` with bit-sliced `__popc` cannot help. Idea rejected on evidence.**

### D3 — where the DFS work actually is (sub100k, sampled 1/128 reads)
Total node-pops **4.09e9 = 40,924/read**. Probes **1.045/pop** (one `2occ4` per node, as designed);
buckets **1.274/probe** (the shared-bucket fast path hits 73% of the time).

Node pops by errors used: `e=0` 0.03% | `e=1` 1.6% | **`e=2` 37.5%** | **`e=3` 59.4%** | `e=4` 1.5%.
Depth mass is at 11-18 (cum 20.7% -> 97.0%), peaking at depth 14. An analytic model
`N(j) ~ [sum_{e<=E(j)} C(j,e) 3^e] x min(1, 6.28e9/4^j)` predicts 41,000 pops/read vs 40,924
measured -- the tree is a thin combinatorial shell, NOT genome-limited, at the depths that matter.

Consequence: k-mer existence bitmaps, ftab/jump tables, L2-resident prefix tables and
singleton-interval collapse all target depths that hold ~no work. **All rejected analytically.**

### D4 — lane utilisation and stack pressure: BOTH FINE (two more ideas rejected)
`waves=1.58e8`, **mean active lanes 25.97/32 = 81%**; `spills=5057` (0.000/wave). The worry that
`n_active = min(sp,32,room/9)` throttles bushy reads to <=14 lanes does not happen: with
CAP_SM=512 the frontier never approaches capacity. No change warranted.

### D5 — THE ACTUAL BOTTLENECK: 320 bytes of local memory per thread
`ptxas -v` on `k_dfs_warp2`: **`320 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`,
80 registers.** Not register spill -- genuine local arrays:
`cck[9]+ccl[9]+ccn[9]` (180 B child buffer, dynamically indexed via `ADD2`) plus
`cntk[4]+cntl[4]` (64 B, passed to `d_bwt_2occ4` by pointer).

The kernel actually runs at 8 warps/SM = 256 threads, so the local working set is 256 x 320 B =
**82 KB**. That is under the 128 KB unified data cache -- but the shared-memory stack already
claims 80 KB of it (2 blocks x wpb=4 x CAP_SM=512 x 20 B), leaving only ~48 KB as L1. So 82 KB of
local traffic contends for ~48 KB of L1 and every node expansion pushes child records out to
L2/DRAM. Invisible in the occ4 accounting, and the missing ~50%.
(An earlier revision of this section said "16 warps/SM, 512 threads, 164 KB vs 128 KB" -- wrong
arithmetic for the shipped configuration; the conclusion holds, the numbers above are correct.)

### Revised plan (evidence-ranked)
1. Eliminate the local-memory child buffer (count -> prefix-sum -> re-emit directly to the
   shared stack, regenerating children from `cntk/cntl` held in registers). No algorithmic change.
2. Two-level 32 B Occ block layout (D2: 1.89x more probes/s). Raises the ceiling once (1) makes
   the kernel actually memory-bound again.
3. Search schemes on the bidirectional index (`bwt.c:262 bwt_extend` works on bwa's existing BWT
   because `T + revcomp(T)` is its own reverse complement). D3 shows `e>=2` is 97% of all pops, so
   capping the first-searched part at `e<=1` removes most of the tree. Estimated 2.5-4x.

### D6 — the occupancy discovery: the "2.32 G-occ4/s HBM wall" is an ARTEFACT of high occupancy
Random 64 B gather over a 3 GB buffer, sweeping resident warps per SM:

| warps/SM | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 64 B probes/s | 5,589 M | **5,844 M** | 4,064 M | 2,584 M | 2,340 M |
| useful GB/s | 358 | **374** | 260 | 165 | 150 |

`fmtest` measured 2.32 G-occ4/s at HIGH occupancy, where the memory system thrashes. At the
DFS kernel's actual occupancy (8 warps/SM, shared-memory limited) the hardware delivers
**5.84 G probes/s = 374 GB/s**, not 143 GB/s. Every "we are at the HBM bandwidth wall"
conclusion in Round 2 was measured against the wrong number.

Same sweep for the Occ primitive itself (3 GB synthetic BWT, matched occupancy, 8 warps/SM):
gather 6.39 G/s | `d_bwt_occ4` (cnt_table) 4.14 G/s | `d_bwt_occ4_s` 4.16 G/s | popc 4.43 G/s.

### Phase 6 RESULTS — 1.82x kernel speedup, bit-exact, no algorithmic change
All three changes are pure code-generation fixes; the visited node set is unchanged.

| change | ptxas stack frame | effect |
|---|---|---|
| baseline (Phase 5) | 320 B | - |
| two-pass child generation (count -> prefix-sum -> emit direct to shared) | 128 B | +21.6% |
| scalar-reference `d_bwt_occ4_s` / `d_bwt_2occ4_s` / `..._alt_s` | **0 B** | +20% |
| `d_occ_aux4` -> bit-sliced `__popc` (default; `-DFM_OCC_CNTTABLE` reverts) | 0 B | +18% |

`cck[9]/ccl[9]/ccn[9]` and every `uint64_t cnt[4]` passed by pointer were address-taken and so
lived in LOCAL memory. At the shipped 8 warps/SM that is 256 x 320 B = 82 KB of local working set
contending for the ~48 KB of L1 left after the 80 KB shared-memory stack, so each node expansion
spilled child records to L2/DRAM -- invisible in the occ4 accounting.
Note `cntk[c]` is indexed by a RUNTIME base `c`, so unrolling alone cannot promote it; it needed
the `SELK/SELL` register select chains.

Interleaved A/B (alternating binaries to cancel thermal drift), bit-exact every run:
- sub100k: 5.4-5.6 s -> 3.2 s = **1.73x**
- 2730 short-read set (671,652 reads, 30-63 bp, mean 50.5): 43.1 s -> 23.7 s = **1.82x**

Occupancy re-swept after the fix: still 8 warps/SM optimal (CAP_SM=512, wpb=4). Raising to
24 warps/SM is SLOWER (spill traffic). `DFS_WARP_WPB` added for future sweeps.

### Correctness / robustness fixes
- **Restored the `GPUALN_HISTO` histogram** (was reporting `reads=0`; `d_npop`/`d_flag` never
  copied back after the Phase-5 refactor).
- **Guarded CAP_SM/CAP_GL.** `CAP_SM=128` made the CHUNK=128 spill path read past the frontier;
  the run died with a CUDA fault yet still left a truncated 357-byte `.sai` on disk (rc=1, so a
  pipeline that ignores exit status could mistake it for output). Now rejected up front.

### Validation at scale (the AVA1B asset is gone; two independent sets used instead)
- **ENNBN7, 3,505,813 reads (30-34 bp, max_diff=3), 23.7% hit-reconciled** -- exercises the CPU
  reconcile path hard. GPU `.sai` **byte-identical** to CPU `bwa aln` (md5 `ff8a202d`).
  CPU 470 s -> GPU 62.4 s = **7.5x**.
- **2730 (AdapterRemoval3-merged pairs, short branch), 671,652 reads (30-63 bp, mean 50.5)** --
  the closest match to the original AVA1B regime. GPU `.sai` **byte-identical** to CPU
  (md5 `aa9302bd`). CPU 235 s -> GPU 24.3 s = **9.7x end-to-end**.

### Where the kernel now stands
5.45e9 buckets over a 3.2 s kernel = ~1.7 G-occ4/s, vs 4.14 G/s for the same primitive at the
same occupancy => still ~2.4x of headroom that is NOT bandwidth, NOT occupancy, NOT the Occ code
and NOT local memory. Remaining suspects are the per-wave serial structure (pop -> probe ->
syncwarp -> prefix-sum -> push, MLP=1 per lane) and warp divergence. Resolving this needs
Nsight Compute (`sudo pacman -S nsight-compute`; not installed).

### D7 — follow-up measurements that CORRECT two claims above (and Idea B's priority)

**(a) Idea B is much weaker than advertised at the real operating point.** The "32 B block =
1.89x more probes/s" figure in D2 was itself measured at HIGH occupancy (82x16 blocks x 128
threads = 32 warps/SM) -- the very artefact D6 identified. Re-measured best-of-3 across occupancy:

| warps/SM | 4 | **8 (shipped)** | 16 | 32 |
|---|---|---|---|---|
| 64 B probes/s | 6,560 M | 6,518 M | 5,681 M | 3,945 M |
| 32 B probes/s | 7,229 M | 7,546 M | 7,592 M | 7,610 M |
| **ratio** | 1.10x | **1.16x** | 1.34x | 1.93x |

At low occupancy the memory system is limited by request rate/latency, not bytes, so halving
bytes-per-probe barely helps; only under high-occupancy thrash does the byte count dominate.
**=> Idea B (two-level 32 B Occ layout) is DOWNGRADED: ~1.16x for a full 3.14 GB index re-layout,
and the kernel is not even memory-bound right now (1.7 vs 4.14 G-occ4/s). Do not build it yet.**

**(b) The popc win is NOT register-pressure relief.** `ptxas -v`: popc = 80 registers,
cnt_table = 79. popc wins *despite* costing one more register, so the mechanism is
constant-cache/instruction cost, matching the isolated occbench result (4.43 vs 4.14 G/s = +7%
at matched occupancy, with the larger +18% in the full kernel coming from the extra constant
traffic there -- `c_L2` plus 536 B of kernel params in cmem[0]).
Caveat on D2: "cnt_table is already fully hidden" was true only of the HIGH-occupancy fmtest
measurement. At 8 warps/SM it was never fully hidden.

## Phase 7 — testing the remaining ideas (E measured, A cost-modelled)

### Idea E — read deduplication: REAL but ~10%, NOT the 1.2-1.5x estimated
`bwa aln` ignores base qualities, so identical sequences give identical trees; dedup + broadcast
of `has_hit` is trivially bit-exact. Measured exact-duplicate rate and the work it actually costs:

| set | reads | distinct | dup rate | node-pops full -> dedup | kernel s full -> dedup |
|---|---|---|---|---|---|
| 2730 short | 671,652 | 603,472 | 10.15% | 46.60e9 -> 41.87e9 (-10.17%) | 24.3 -> 23.1 (-4.9%) |
| ENNBN7 | 3,505,813 | 3,136,527 | 10.53% | 94.66e9 -> 84.86e9 (-10.35%) | 63.5 -> 57.4 (-9.6%) |

**The "duplicates are the bushy low-complexity reads, so the saving is superlinear" hypothesis is
FALSE.** Work saved (10.17% / 10.35%) tracks the read-count reduction (10.15% / 10.53%) almost
exactly, even though ENNBN7's top duplicates are poly-CA/poly-TG (x84, x79). Those reads do not
have oversized trees.
**Verdict: ~10% of GPU work, 5-10% wall. Worth doing, but it is a 10% lever, not a 1.5x one.**

### Idea A — search-scheme cost, MEASURED (not modelled)
Added `DFS_STAIR=p:u1:u2` to the instrumented binary (`dfs_engine.cuh`, `STAIR_OK`): children whose
error count exceeds the staircase budget U(depth) are never pushed, so `g_pops` reports that
search's TRUE tree size. (A single staircase search is not a superset on its own -- the `.sai`
from such a run is meaningless; this measures cost only.)

Cost of ONE search, as a fraction of the current full d=4 backtrack:

| p:U1:U2 | sub100k | 2730 short |
|---|---|---|
| 2:2:4 | 44.59% (2.2x) | 26.89% (3.7x) |
| 3:1:2 | **7.09% (14.1x)** | **2.76% (36.2x)** |
| 3:1:3 | - | 5.67% (17.6x) |
| 4:1:2 | 19.87% (5.0x) | 8.66% (11.6x) |
| 5:0:1 | 3.34% (30.0x) | 1.01% (99.3x) |

Consistency check: the 2-part `2:2:4` figure on sub100k (44.6%) matches the independent D3
prediction from the error histogram (`e<=2` = 39% of pops). Model and instrumentation agree.

**Why the union of searches is exact, not merely a filter:** a *covering* search scheme has, for
every distribution of the k errors across the p parts, at least one search whose staircase admits
it. So the union finds every occurrence with <= max_diff edit ops -- no false negatives -- while
each individual search only ever reports genuine <= max_diff occurrences. Against bwa the union is
a strict superset (bwa additionally restricts via `indel_end_skip` / `max_del_occ` / `max_gapo`),
which is exactly the safe direction: false positives fall through to the existing exact CPU
reconcile.

**Total cost = (number of searches) x (per-search cost).** A covering scheme for k=4 needs roughly
4-8 searches, so: 3-part at 7.09% x 6 = 43% (**2.3x**); 5-part at 3.34% x 6-8 = 20-27% (**3.7-5x**).
**Realistic range 2.5-5x** -- this CONFIRMS the revised 2.5-4x estimate and REFUTES the original
optimistic "13x" (that figure costed a single search, not a covering set).

Unmeasured risk: searches anchored at a NON-terminal part need true bidirectional extension, and
their cost profile is not captured by this unidirectional probe. They are likely dearer than the
3'-anchored searches measured here, so treat 2.5-5x as an upper-ish bound until built.

### Status of the remaining levers
| idea | estimate before | measured | verdict |
|---|---|---|---|
| A search schemes | 3-13x | 2.5-5x (per-search cost measured) | **headline lever, unbuilt** |
| B 32 B Occ layout | 1.5-2x | 1.16x at shipped occupancy | shelved (D7) |
| C popc | 1.1-1.2x | +18% | DONE, shipped |
| D proactive spill | 1.0-1.1x | 81% lanes, 0 spills | rejected (D4) |
| E read dedup | 1.2-1.5x | ~10% work, 5-10% wall | real but small, unbuilt |
| F CPU 17th worker | +15% | ~+5-9% (GPU:CPU now ~10:1) | unbuilt |

## Phase 8 — Idea A foundations: device bidirectional FM-index + anchor cost probe

### Device bidirectional extension — BUILT and VALIDATED
`fm_device.cuh`: `bwtintv_dev {x0,x1,x2}`, `d_bwt_extend()`, `d_bwt_set_intv()` -- device mirrors of
bwa's own `bwt_extend()` (bwt.c:262) and `bwt_set_intv()`. Works on the EXISTING `.bwt` with no
second index and no extra memory, because S = T . revcomp(T) satisfies revcomp(S) = S.
Semantics (from `bwt_smem1a`): `x0` = SA interval of P, `x1` = interval of revcomp(P), `x2` = size;
`is_back=1` prepends, `is_back=0` appends and must select `ok[3-c]` (the complement).

`fmtest` Test C replays a random alternating forward/backward walk on device and on host with
bwa's `bwt_extend`, comparing the full triple at every step:
**PASS -- 20,000 queries, 300,630 steps, 0 mismatches.** (All queries die after ~15 steps, matching
log4(6.28e9) = 16.3 -- an independent check of the genome statistics.)

### `bidir_cost` — the anchor question, ANSWERED
New tool `cuda/bidir_cost.cu` (`make bidir_cost`): one staircase search, warp-cooperative, with a
configurable ANCHOR position; reports node-pops. Mismatch-only, no width bound -- a COST PROBE.

Phase 7's `DFS_STAIR` could only price searches anchored at the 3' END. A real search scheme also
needs searches anchored at INTERIOR parts (true bidirectional), and their cost was the single
unmeasured term in the 2.5-5x estimate. Measured (sub100k, max_err=4, p=3, U=(1,2,4)):

| anchor | 5' end | 1/4 | middle | 3/4 | 3' end |
|---|---|---|---|---|---|
| pops/read | 853 | 851 | 854 | 856 | 856 |

**Anchor position is irrelevant (<=0.4% spread).** Pure-forward, bidirectional and pure-backward
searches all cost the same: tree size is set by the staircase budget and genome statistics, not by
direction. Holds across every config tried (spread 1-3%):

| config (mismatch-only) | mean pops/read |
|---|---|
| p=4 U=(0,1,4) | 97 |
| p=3 U=(1,2,4) | 854 |
| p=5 U=(0,1,4) | 880 |
| p=4 U=(1,2,4) | 2,381 |
| p=5 U=(0,2,4) | 3,717 |
| p=3 U=(4,4,4) (= full, no staircase) | 105,030 |

**CAVEAT -- do not quote these as scheme speedups.** This probe has no gaps and, more importantly,
no `bwt_cal_width` admissible bound, so its "full" baseline (105,030 pops/read) is 2.6x LARGER than
the production baseline (40,924, which has both). Ratios taken against it are inflated. The
authoritative per-search costs remain the Phase-7 `DFS_STAIR` numbers measured inside the real
engine (p=3: 7.09% of production; p=5: 3.34%). `bidir_cost` contributes the anchor RATIO, which is
measured within a single probe and is therefore valid regardless.

### Scheme design is the real remaining work (and it is not a simple staircase)
A monotone staircase is a cost model, not a covering scheme. Worked counter-example for p=3, k=4,
U=(1,2,4): the error distribution (0,4,0) is admitted by NO search -- anchoring at part 1 or 3 hits
the cumulative cap of 2 at the second part, and anchoring at part 2 needs e2<=1. Parts must
therefore satisfy p >= 5 for k=4 (pigeonhole: some part is error-free), and even then a plain
staircase blocks distributions like (0,4,0,0,0) because pi must stay CONTIGUOUS, so the heavy part
cannot always be deferred to last. This is exactly why the literature solves scheme design with a
MIP (Kianfar/Pockrandt; Renders/Fostier). **Next step is to adopt a published covering scheme for
k=3/4/5 and price it with the per-search costs above, not to invent one.**

Revised estimate unchanged: **2.5-5x**, now with the anchor risk retired.

## Phase 9 — Nsight Compute: the ~2.4x gap explained, and longest-first scheduling

`ncu` needed two attempts. The first (`--replay-mode kernel`, the default) produced a report with
`Invocations 1` but every metric NaN and grid `(0,0,0)`: kernel replay must snapshot/restore all
device memory the kernel can write, and this kernel sits on a 3.14 GB BWT + 215 MB backing.
**Use `--replay-mode application` for this kernel** (re-runs the binary per pass instead).

### Profile of `k_dfs_warp2` (sub100k, post-Phase-6 engine)
| metric | value | reading |
|---|---|---|
| DRAM throughput | **9.65%** | NOT bandwidth-bound. D6 confirmed by the hardware counters. |
| Compute (SM) throughput | 16.74% | not compute-bound either |
| No eligible warp | **74.7%** of cycles | schedulers starved |
| Active warps / scheduler | 1.96 of 12 | = the 8 warps/SM the shared stack allows |
| Stall `long_scoreboard` | **49.1%** of 7.8 warp-cycles | memory dependency, MLP = 1 probe per lane |
| Avg active threads / warp | **10.81 / 32** | heavy instruction-level divergence (est. 11% available) |
| SM active / elapsed cycles | 3.21e9 / 5.01e9 = **64%** | **36% of the kernel is straggler tail** |

So the missing ~2.4x is three things, none of them bandwidth: (a) memory latency that cannot be
hidden with only ~2 warps/scheduler and one outstanding probe per lane; (b) divergence -- only
10.8 of 32 threads active per instruction, even though 25.97/32 lanes pop a node (D4); the loss is
in child generation and the occ4 word-scan, not the pop; (c) a straggler tail.

Note (b) reconciles with D4: 81% lane utilisation at the POP, 34% averaged over all instructions.

### Longest-first work-pool scheduling — SHIPPED, bit-exact
Tree size grows exponentially with `max_diff`, which is a step function of read length, so handing
out the longest reads FIRST keeps the tail cheap (classic LPT scheduling). Implemented as an
`order[]` permutation indirection in the work pool (`dfs_engine.cuh`); scheduling order only
decides which warp takes which read, so `has_hit` and the `.sai` are unchanged.
A/B knob: `GPUALN_NOORDER=1`.

| dataset | length range | unordered | longest-first | speedup |
|---|---|---|---|---|
| sub100k | 30-91 bp (max_diff 3-6) | 2.7 s | **1.8 s** | **1.50-1.59x** |
| 2730 short | 30-63 bp | 23.6 s | 22.0 s | 1.07x |
| ENNBN7 | 30-34 bp (all max_diff=3) | 62.7 s | 63.1 s | 0.99x (noise) |

**The gain is proportional to read-length VARIANCE** -- a uniform-length library has no tail to
fix. Never negative, so it is on by default. Caveat for the production pipeline: `adna_aligner.sh`
splits at 64 bp, so the GPU normally sees 30-63 bp (the ~1.07x case). The 1.5x applies to
unsplit/wide-spread input (e.g. `-A`, a higher cutoff, or the original AVA1B 30-91 bp profile).

**Cumulative vs the pre-Phase-6 engine (sub100k): 2.56-2.61x, bit-exact (md5 eecf35c1).**

### What remains of the gap
Still open: (a) MLP -- process 2+ nodes per lane per wave so several independent probes are in
flight before the warp stalls; directly targets the 49.1% `long_scoreboard`. (b) Divergence at
10.8/32. (c) Occupancy is only 1.96 warps/scheduler, capped by the 40 KB/block shared stack; worth
re-sweeping now that local memory is gone (the last sweep predates Phase 6).

### MLP per lane (2 nodes/lane/wave) — IMPLEMENTED, MEASURED, REVERTED (negative result)
Phase 9's ncu profile pointed at memory latency (49.1% `long_scoreboard`, 74.7% of cycles with no
eligible warp) and, since occupancy is capped at 8 warps/SM and D6 shows the memory system peaks
there, the indicated fix was more in-flight probes per LANE. Built it: each lane pops NPL=2 nodes
into two register-only slots (A, B) and issues both `d_bwt_2occ4` calls before consuming either
result. Token-pasted slot macros (`DECL_SLOT`/`LOAD_SLOT`/`PREP_SLOT`/`SELK(S,x)`) keep every count
in named registers -- `ptxas` confirmed **0 bytes stack frame**, so the Phase-6 local-memory fix
was not regressed. Registers 79 -> 127, which is free (8 warps/SM = 256 threads/SM allows 256).

**Result: 0.90-0.95x — SLOWER. Reverted.**

Why, measured: with capacity for 64 nodes per wave the engine still pops only **28.35 per wave**
(instrumented; NPL=1 pops 25.97). Slot B is almost always empty.

**Root cause is structural: the DFS branching factor is ~1** (4.09e9 pops vs 4.09e9 pushes), so a
wave pops n and pushes ~n and the frontier sits at an equilibrium of ~28 nodes. Popping more per
wave drains the frontier faster and it self-limits. **Per-lane MLP cannot be raised this way, at
any NPL.** The wave is already at 89% of the frontier's width, not starved of lanes.

Consequence: raising in-flight probes per warp requires a WIDER frontier, which means more than one
read per warp (each read contributes an independent ~28-wide frontier). That needs 2 stacks/warp:
either CAP_SM 512->256 per read (the sweep says slower -- more spills) or double the shared memory
(halves occupancy to 4 warps/SM). Neither is attractive, so **this direction is closed** unless the
node is shrunk enough to afford two stacks at 8 warps/SM.

Restored engine: 80 registers, 0 B stack frame, 1.8 s, md5 `eecf35c1`.

## Phase 10 — Idea A step 1: the width-bound risk is RETIRED

`IDEAS_A_E_F.md` §1.7.1 flagged the loss of bwa's `bwt_cal_width` bound as the largest open risk in
Idea A: that bound is *directional* (backward search only), so interior-anchored searches cannot
use it, and the probe suggested it was worth ~2.6x.

### The bound generalises to two sides
At a node with window `[lo,hi)` the unmatched flanks are `seq[0..lo-1]` and `seq[hi..len-1]`.
Errors in disjoint segments add, so `bidL[lo] + bidR[hi]` is admissible, where
`bidL[x]` = min errors for `seq[0..x-1]` and `bidR[x]` = min errors for `seq[x..len-1]`.
Justification is bwa's own: if a string occurs with e errors, deleting the error positions leaves
e+1 error-free pieces that each occur in the genome, and greedy maximal chopping is minimal, so
(pieces-1) <= e. Each array is ONE O(len) greedy pass on the bidirectional index -- `bidL` by
appending (forward extension), `bidR` by prepending. Implemented in `bidir_cost.cu`
(`d_build_bounds`, lane 0 and lane 1 in parallel); enable with the 7th argument.

### Measured (sub100k, max_err=4, pops/read)
| config | anchor | bound OFF | bound ON | gain |
|---|---|---|---|---|
| full search U=(4,4,4) | 5' end | 102,049 | 26,879 | 3.80x |
| full search | 1/4 | 103,811 | 72,385 | **1.43x** |
| full search | middle | 104,548 | 41,295 | 2.53x |
| full search | 3' end | 109,274 | 27,025 | 4.04x |
| **staircase U=(1,2,4)** | 5' end | 853 | 820 | **1.04x** |
| **staircase U=(1,2,4)** | middle | 854 | 831 | **1.03x** |
| **staircase U=(1,2,4)** | 3' end | 856 | 823 | **1.04x** |

(Caveat: the full-search rows are truncated lower bounds -- 88,410 of 99,166 reads overflow the
CAPSM=384 probe stack without the bound. The staircase rows are clean, <=7 overflows.)

### Verdict
Two findings, and the second is what matters:
1. For an UNPRUNED search the bound is worth 3.8-4.0x at end anchors but only **1.43x at an
   interior anchor** -- exactly the predicted mechanism (two short flanks each have a small `bid`;
   one long flank has a large one). So the concern was real *for unpruned searches*.
2. **With a staircase budget the bound is worth 3-4%, at every anchor.** The staircase already
   prunes what the bound would, and anchor-independence still holds (820-840, 2.4% spread).

**=> A search-scheme engine does not need bwa's width bound at all, and interior anchors are not
penalised. The largest open risk in Idea A is retired.** No second `bwt_cal_width` pass is needed
in preprocessing either.

Per-search cost estimates are unchanged and remain the Phase-7 `DFS_STAIR` figures measured in the
real engine (with gaps + bwa's bound): 7.09% of production baseline at p=3, 3.34% at p=5.
Since p>=5 is required for k=4 (a plain staircase at p=3 is not covering), the scheme estimate is
5-8 searches x 3.34% = 17-27% => **3.7-5.9x**; overall band still **2.5-5x**.

**Next: adopt a published covering scheme for k=3/4/6 (step 2).**

## Phase 11 — Idea A step 2: published schemes implemented and PRICED. Verdict ~2.5-3x, not 3-13x

Paper in-tree: `cuda/1711.02035v2.pdf` (Kianfar, Pockrandt, Torkamandi, Luo, Reinert).
Table 3 transcribed verbatim into `cuda/bidir_cost.cu`; Table 1/2 read for context.

### Full (pi, L, U) semantics implemented
`k_scheme` in `bidir_cost.cu` runs ONE search of a scheme with the real semantics:
- parts are equal-size (the paper's MIP optimises over equal pieces);
- `U[j]` (cumulative upper bound) enforced continuously inside part `pi[j]` -- safe, since the
  error count only grows;
- `L[j]` (cumulative LOWER bound) checked on COMPLETING a part. **L is what stops the searches of
  a scheme redoing each other's work; omitting it makes a scheme look ~2x more expensive.**
- `pi` is contiguous, so the window `[lo,hi)` always extends into an adjacent part; direction is
  "grow right if `tgt_hi > hi`, else grow left if `tgt_lo < lo`".

**Covering verified exhaustively** for all 7 transcribed schemes (K=1..4, P=K+1 and K+2): every
error distribution with sum <= K is admitted by some search, and every `pi` is contiguous.
0 uncovered out of 3/10/35/126/15/56/210 distributions. This is the unit test that guards against
silent false negatives; it must be re-run if a scheme is ever edited.

### Cost, measured on sub100k (mismatch-only, node-pops)
Against backtracking **without** the width bound (the naive comparison):
| K | P=K+1 | P=K+2 |
|---|---|---|
| 3 | 25.6x | 48.5x |
| 4 | 9.0x | 8.8x |

Against backtracking **with** the two-sided bound on BOTH sides (the production-relevant one,
since the shipped engine already has bwa's bound):
| K | P=K+1 | P=K+2 |
|---|---|---|
| 3 | 1.40x | **2.97x** |
| 4 | 2.55x | **2.70x** |

**The 9-48x figures are inflated: the width bound alone supplies much of the pruning a scheme
supplies, so the two overlap heavily and must not be counted twice.** Only the bound-on-both-sides
column is meaningful for this codebase.

### Verdict on Idea A
**~2.5-3.0x on node-pops, in the best case**, using P=K+2 schemes (K=3 -> 2.97x, K=4 -> 2.70x).
Further discounts to expect before it becomes wall-clock:
1. **Hamming vs edit.** These numbers are mismatch-only. bwa needs edit distance, and the paper's
   own Table 2 shows edit gains are ~0.65x of Hamming (K=3: 20.8x edit vs 32.4x Hamming). Expect
   nearer **~2x**.
2. Node-pops is a good proxy here (the kernel is probe-bound) but the bidirectional node is 24 B
   vs 20 B and carries extra L/U bookkeeping and register pressure.
3. Our regime is NOT the paper's. Table 2's headline speedups are R=101 with K<=3; we are R=30-63
   with K=3-4. Table 1 already shows K=4 edge reduction is only a factor 0.59, and Table 2 has no
   K=4 row at all.

So the original "3-13x" was far too optimistic; the revised "2.5-5x" was about right at its lower
bound. **Realistic: ~2x wall-clock for a substantial engine rewrite**, versus the 2.56-2.61x
already banked this round from code-generation and scheduling fixes alone.

## Phase 12 — Renders schemes REVERSE the Idea A verdict (14.7-16.6x, not 2.5x)

`cuda/IDEAS_RESEARCH_2026.md` (external literature scan) §2.1 argued that Kianfar's k>=3 schemes
are the wrong ones for this workload -- they use few searches but each ends at terminal U=k, giving
"heavy tails", whereas Renders et al. and SeqAn3 keep the first-searched part at L=U=0/1 and defer
the full budget to the last, already-narrowed part. Since this kernel's cost IS the e>=2 shell
(97% of pops, D3), that shape should matter enormously. **Tested: it does.**

Transcribed and covering-verified (exhaustive, contiguity + all error distributions):
SeqAn3 `optimum_search_scheme<0,3>` (k=3, p=5, 4 searches) and Renders `multiple_opt/4` scheme1
(k=4, p=5, 5 searches). Both added to `cuda/bidir_cost.cu` (selected as `P=105`).

### Cost vs backtracking, two-sided bound on BOTH sides (node-pops, mismatch-only)
| dataset | K | Kianfar | SeqAn3 / Renders |
|---|---|---|---|
| sub100k | 3 | 1.40x (p=4) / 2.97x (p=5) | **3.30x** (SeqAn3 p=5) |
| sub100k | 4 | 2.55x (p=5) / 2.70x (p=6) | **14.69x** (Renders p=5) |
| 2730 short | 3 | 1.42x | **3.82x** (SeqAn3) |
| 2730 short | 4 | 2.13x | **16.62x** (Renders) |

### Executor validated against backtracking (the check that matters)
Added a per-read hit-set dump (`BIDIR_DUMP=file`). A covering scheme must find EXACTLY the reads a
plain unrestricted search finds. Restricting the comparison to reads where neither side overflowed
the probe stack:

| scheme | clean reads compared | FALSE NEGATIVES | extra |
|---|---|---|---|
| Kianfar p=5 | 94,271 | **0** | 0 |
| Renders p=5 | 94,293 | **0** | 0 |

Both schemes find identical hit sets (592 reads each). Backtracking's lower raw count (160) is
purely its 4,868 stack overflows aborting searches early -- the schemes overflow on 276 and 11.
So the executor is correct and the covering property holds empirically as well as combinatorially.

### Verdict — Phase 11's "~2x, don't build it" is WITHDRAWN
That recommendation was measured on **Kianfar's** tables, which are the wrong family for k>=3
(Kianfar's k=3/k=4 entries are explicitly "best solution found in 2 hours", not proven optimal;
SeqAn3 ships `// TODO computation has not finished` for k=4). With the Renders/SeqAn3 tables the
same engine, same probe, same baseline gives **14.7-16.6x at K=4 and 3.3-3.8x at K=3**.

Remaining honest discounts before this is wall-clock:
1. mismatch-only; production allows gaps (the probe's bound-enabled backtracking is 26,879
   pops/read vs production's 40,924 with gaps -> ~1.5x inflation on both sides);
2. Hamming tables + edit-distance slack (the community practice -- Columba does exactly this, and
   it only adds false positives, which this architecture absorbs via the CPU reconcile). The
   paper's Table 2 suggests edit gains ~0.65x of Hamming;
3. production reads are 30-63 bp -> a mix of K=3 (3.3-3.8x) and K=4 (14.7-16.6x), so the weighted
   figure depends on the length distribution (2730 short, mean 50.5 bp, is mostly K=4);
4. node-pops is a good proxy (the kernel is probe-bound) but the bidirectional node is 24 B vs
   20 B with extra L/U bookkeeping.

**Even after those, Idea A is now clearly the largest remaining lever and worth building.**
Use Renders (k>=3), not Kianfar. Kianfar remains fine for k=1/2.

### Also tested from the same document
- **`__ldcs` streaming hint** on all 18 BWT loads (§6): **no change** (1.7-1.8 s either way,
  md5 `eecf35c1`). `__ldg` already routes through the read-only path on Ampere and the kernel is
  not L2-capacity-limited. Rejected.
- **§3 per-lane MLP / prefetch** ("the single most important new lever"): already built and
  REJECTED in Phase 9 at 0.90-0.95x. The document predates that measurement. The frontier
  self-limits at ~28 nodes (branching factor ~1 -> pops == pushes), so there is no node i+1 to
  prefetch; QuadRank's 2x comes from batching *independent* rank queries, whereas ours are
  dependent by construction. Only the read-batching variant (§3.3) survives, and it needs 2 stacks
  per warp -- costing either occupancy or spills.
- **§9 step 1 "measure the width-bound loss, still the biggest unknown"**: done in Phase 10 and
  retired -- with a staircase in place the bound is worth 3-4%.

## Phase 13 — SEARCH-SCHEME ENGINE SHIPPED: 20x kernel, byte-identical

`cuda/schemes.cuh` + `cuda/scheme_engine.cuh`, wired into `aln_gpu.cu` behind `GPUALN_SCHEME=1`.

### What it is
Replaces the monolithic max_diff-deep backtrack with a covering search scheme run on the
bidirectional FM-index (bwa's own `.bwt`; no second index, no extra GPU memory).

- **Tables** (`schemes.cuh`): K=1,2 Kianfar; **K=3 SeqAn3** (p=5, 4 searches); **K=4 Renders**
  (p=5, 5 searches). Every table is **covering-verified exhaustively at startup** -- contiguity of
  `pi` plus all `(K+1)^P` error distributions. A non-covering table causes silent false negatives,
  so this check is a hard gate, not a debug aid.
- **Fallback**: reads with K>=5 or len>127 run the existing exact engine. For the production short
  branch (L<=63 => K=3,4 only) the fallback is never taken.
- **Node**: 24 B (`x0,x1` u64 + `x2,packed` u32). `x2` fits u32 because the largest single-base
  interval is ~1.57e9 and intervals only shrink. At CAP_SM=512 that is 48 KB/block, 2 blocks/SM --
  **8 warps/SM preserved**, and `ptxas` reports **0 bytes stack frame** (no local-memory regression).
- **Edit distance** needs no boundary slack: part membership is defined by READ position, so every
  edit alignment induces a well-defined error distribution over parts and a covering scheme admits
  it. A deletion (no read base consumed) is attributed to the part being extended.
- **`SST_E` (EMPTY) state**: a search must be able to represent an insertion at *its own seed
  position*, so the seed is an empty pattern whose children are the 4 single-base seeds plus an
  insertion. Without this a search could miss alignments -- a false negative.
- Keeps bwa's `max_gapo`/`max_gape` (bwa's own alignments obey them); **drops `indel_end_skip` and
  `max_del_occ`**, which only ever remove candidates -> strictly more permissive -> superset.

### Measured
| dataset | reads | L (K) | fallback | kernel exact -> scheme | end-to-end | vs CPU bwa aln |
|---|---|---|---|---|---|---|
| 2730 short | 671,652 | 30-63 (3,4) | **0%** | 22.2 s -> **1.1 s = 20.2x** | 24.3 -> 4.6 s (5.3x), 145,532 r/s | 235 s => **51x** |
| ENNBN7 | 3,505,813 | 30-34 (3) | **0%** | 65.3 s -> ~ | 62.4 -> 19.9 s (3.1x), 175,789 r/s | 470 s => **23.6x** |
| sub100k | 100,000 | 30-91 (3-6) | 10.3% | 1.8 s -> 1.4 s = 1.29x | 2.3 s, 43,198 r/s | - |

sub100k gains least because 10.3% of its reads are K>=5 and fall back to the exact engine -- and
those are the longest reads with the largest trees, so they dominate what remains. This is the
unsplit case; the production pipeline splits at 64 bp and sees 0% fallback.

### Bit-exactness
| set | result |
|---|---|
| sub2k | `.sai` md5 `4b068014` == golden |
| sub10k | `.sai` md5 `54410c75` == golden |
| sub100k | `.sai` md5 `eecf35c1` == golden |
| 2730 short (671,652) | **byte-identical to CPU `bwa aln`** |
| ENNBN7 (3,505,813) | **byte-identical to CPU `bwa aln`** |

~4.3 M reads across three independent datasets, all byte-identical. Flag rate rises only slightly
(2.619% -> 2.671% on 2730; 23.673% -> 23.755% on ENNBN7), i.e. the superset costs a handful of
extra CPU reconciles, exactly as designed.

### The bottleneck has moved to the CPU reconcile
ENNBN7 now spends 17.8 s of CPU reconcile against a ~20 s total: 832,822 of its reads are hits and
every hit is re-aligned by `bwt_match_gap` on the CPU. **That is why its end-to-end gain (3.1x) is
far below its kernel gain.** For hit-heavy libraries the next lever is no longer the GPU:
- Idea F (CPU as an extra aligner) is now backwards -- the CPU is the *bottleneck*, not idle;
- worth attacking instead: reconcile only what the `.sai` truly needs, or move the reconcile's
  `bwt_match_gap` onto the GPU for the flagged subset.
For the 0.5%-mapping aDNA libraries this project targets (AVA1B, 2730), the reconcile is small and
the 20x kernel gain carries through.

### Status
`GPUALN_SCHEME=1` is opt-in. Given ~4.3 M reads of byte-identical validation plus the startup
covering gate and the automatic K>=5 fallback, promoting it to default is justified; left opt-in
pending a decision.

## Phase 14 — CompileIQ ptxas tuning: 1.17x on the exact engine, ~1.04x on the scheme engine

`cuda/tune_compileiq.py` drives NVIDIA CompileIQ over the undocumented ptxas Advanced Controls.
Objective = GPU-kernel seconds (best of N runs, to suppress the thermal drift that plagued every
measurement this round); **hard correctness gate**: any config whose `.sai` md5 differs from the
golden value scores INVALID, so a fast-but-wrong schedule can never win. `num_workers=1` so GPU
timings never contend.

Run:  `/home/teemu/sorsa/CompileIQ/.venv/bin/python cuda/tune_compileiq.py [--scheme] [--reads F]
       [--golden MD5] [--generations G] [--pool P] [--repeats R]`
Apply: `nvcc ... -Xptxas=--apply-controls=cuda/gpualn.acf`

### Results (6 generations x pool 20, ~120 configs, ~25 min each)
| target | baseline | best | tuner | independent interleaved A/B |
|---|---|---|---|---|
| `k_dfs_warp2` (exact engine, sub100k) | 1.813 s | 1.509 s | 1.201x | **1.165 / 1.172 / 1.186x** -> ~1.17x |
| `k_dfs_scheme` (scheme engine, 2730 short) | 1.143 s | 1.100 s | 1.039x | 1.040 / 0.985 / 1.091 / 1.044x -> ~1.04x |

All runs bit-exact (`eecf35c1`, `aa9302bd`).

**The exact-engine gain is real** (tight 1.165-1.186 band, no overlap with baseline).
**The scheme-engine gain is marginal** -- one of four runs came in below 1.0 and the ranges
overlap, so ~4% is at the edge of measurability here. Do not quote it as a speedup.

Note the tuner's own figures are slightly optimistic (1.201x vs 1.17x measured) because it scores
best-of-N; always re-verify a winning ACF with an interleaved A/B.

### Two findings worth keeping
1. **ACFs do not transfer between kernels.** The ACF tuned for `k_dfs_warp2` gives ~1.17x there
   but 0.974-1.014x on `k_dfs_scheme`. Tune the engine you intend to SHIP, and use a read set with
   0% fallback (L<=63) or the objective silently mixes both kernels. `--reads`/`--golden` exist for
   this.
2. **The winning schedule REINTRODUCED 40 B of stack frame and +9 registers** (on `k_dfs_scheme`;
   `k_dfs_warp2` stayed at 0 B / 89 regs). Removing local memory was worth 40% in Phase 6, yet here
   ptxas trades a little of it back for a better schedule and still wins -- the codegen landscape
   is not monotone in any single `ptxas -v` metric, which is exactly why an empirical search beats
   hand-tuning flags.

Caveat for reuse: an ACF is pinned to the ptxas version it was searched against (13.3 here).
Re-run after a CUDA upgrade rather than assuming it still holds.
