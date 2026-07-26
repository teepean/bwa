# External research — additional speedup levers (literature scan, 2026)

Scope: a web/literature scan (CUDA 13.x docs, PTX ISA, NVIDIA best-practices, arXiv /
ACM / IEEE 2020–2026, GitHub) for optimisation ideas *beyond* the three already costed in
`IDEAS_A_E_F.md`. Nothing here has been measured on this kernel; estimates are from the
cited sources, translated to this workload. Authoritative measurements remain in
`PROGRESS.md`. This document explains what the literature offers, what is actionable, and
what is provably dead — so the dead ends are not re-explored.

> **STATUS AFTER MEASUREMENT (see `PROGRESS.md` Phases 9-12).** Three items here are now settled:
> - **§2.1 (Renders beats Kianfar for this workload) — CONFIRMED, and it is the big one.** Measured
>   14.69x / 16.62x at K=4 vs Kianfar's 2.55x / 2.13x, and 3.30x / 3.82x at K=3. Executor validated
>   against backtracking: 0 false negatives over ~94k clean reads. This REVERSES the earlier
>   "~2x, don't build it" verdict, which had been measured on Kianfar's tables.
> - **§3 (per-lane MLP / prefetch) — REFUTED.** Built and measured at 0.90-0.95x. The frontier
>   self-limits at ~28 nodes (branching factor ~1), so there is no node i+1 to prefetch. QuadRank's
>   2x batches *independent* rank queries; ours are dependent by construction. Only §3.3
>   (read-batching) survives, and it costs occupancy or spills.
> - **§6 `__ldcs`** — measured, no change; rejected.
> - **§9 step 1 (width-bound loss)** — done in Phase 10, retired (worth 3-4% once a staircase exists).

Two findings dominate everything else:

1. **The published, MILP-optimal search-scheme tables that Idea A needs already exist and
   are free** (Columba / Renders et al., k=2/4/6; SeqAn3, k≤3). They are extracted verbatim
   in §A. This de-risks "Idea A step 2 — adopt a published covering scheme" from a research
   task to a transcription + offline-coverage-check task.
2. **The ~2.4x kernel↔primitive gap (1.7 vs 4.14 G-occ4/s) has a known fix that does not
   need Nsight Compute first**: raise per-lane memory-level parallelism (software-pipelined
   prefetch / read-batching). QuadRank (SEA 2026) measured **2x from prefetch alone on
   exactly bwa's rank structure**, and the fix is bit-exact here because `has_hit` is
   order-independent. This is the single most important *new* lever the scan found.

Ranked summary (everything is multiplicative with Idea A):

| # | lever | est. | bit-exact | built? | section |
|---|---|---|---|---|---|
| 1 | **A — search schemes** (now with concrete tables) | 2.5–5x | yes (superset) | foundations only | §2, §A |
| 2 | **per-lane MLP / prefetch** (close the 2.4x gap) | up to ~2x of the gap | yes (order-free) | no | §3 |
| 3 | **k-step FM-index, start k=2** | 1.2–1.4x | yes (exact LF) | no | §4 |
| 4 | **CompileIQ** ptxas advanced-control search | single-digit–~2x swings seen | yes | no | §5 |
| 5 | read-batching across reads (de-divergify + MLP) | folds into #2 | yes | no | §3.3 |
| 6 | `__ldcs` streaming hint on once-touched BWT | 0–5% | yes | no | §6 |
| 7 | multi-GPU batch queue (3090 + A4000) | ~1.5x w/ 2nd GPU | yes | scaffolded | §7 |
| — | E (dedup), F (CPU worker) | ~10%, +8–13% | yes | no | `IDEAS_A_E_F.md` |

Rejected on evidence (do not re-explore): L2 persistence / access-policy window, `cp.async`
for the gather, warp specialization, raising occupancy, 32 B / compressed Occ layouts,
popcount/broadword tricks, caching/tiling, checkpoint re-spacing, dynamic parallelism,
cluster launch control, Blackwell work-stealing. Reasons in §8.

---

## 1. What CUDA 13.x actually gives sm_86

**Nothing new.** The CUDA 13.3 release notes list "None" under new CUDA-platform / CCCL /
CUDA-Python features; every marquee 13.x performance win is gated to Hopper/Blackwell
(cuBLAS TMA SYMV "Hopper and newer", cuSOLVER `Xgetrf` for `sm_90/100/103/120`, FP4/FP8
grouped GEMM on cc 10.x/11.0, PDL "sm_90 and above"). The PTX bulk/tensor-copy family
(`cp.async.bulk`, TMA), cluster launch, distributed shared memory, `wgmma` and tcgen05 are
**all sm_90+ and unavailable on sm_86**. The sm_86-relevant knobs this project cares about —
`prefetch`/`prefetchu` (sm_20+), the `.level::prefetch_size` load qualifier (sm_75+),
cache eviction-priority hints (sm_70+), non-bulk `cp.async` (sm_80+), L2 access-policy
(cc 8.0+) — **all predate CUDA 13**.

> **Consequence:** "upgrade to CUDA 13" buys this kernel exactly one thing — a newer `ptxas`
> instruction scheduler. That reframes CompileIQ (§5) as the *only* genuinely
> CUDA-13-specific path. Source: CUDA 13.3 release notes
> (docs.nvidia.com/cuda/cuda-toolkit-release-notes), PTX ISA 9.3
> (docs.nvidia.com/cuda/parallel-thread-execution).

---

## 2. Idea A is now a transcription task (the search-scheme tables exist)

`IDEAS_A_E_F.md` §1.5 concluded: "the next step is to adopt a published covering scheme for
k=3/4/6 … not to invent one." The scan confirms those schemes are published **as plain-text
`pi/L/U` files**, in exactly this engine's format (0-based, contiguous `pi`, monotone L/U),
inside the **Columba** repository (`github.com/biointec/columba`, `search_schemes/`). They
come from:

- **Kianfar, Pockrandt, Torkamandi, Luo, Reinert** — MILP-optimum schemes (arXiv:1711.02035,
  RECOMB-Seq 2018 / Pockrandt FU-Berlin thesis 2019). Optimal *proven* only for k=1,2;
  k=3,4 are best-known. ~35x over backtracking at 101 bp / k=2.
- **Renders, Depuydt, Rahmann, Fostier** — greedy + ILP solving optimum schemes **to k=7**
  (J. Comput. Biol. 31(10):975–989, 2024, doi:10.1089/cmb.2024.0664; RECOMB 2024). Tool =
  **hato** (`github.com/biointec/hato`, CPLEX, AGPL). Executor = Columba. Ships **co-optimal
  variants per k for dynamic selection** (up to 53% runtime reduction at high k).
- **Gottlieb & Reinert** — a closed-form construction (k+1 searches, p=k+2) that matches
  optimum for k≤3 and beats every known scheme for k≥4, verified valid+complete to k=15
  (NAR Genomics Bioinformatics 7(1):lqaf025, 2025, PMC11915513). Also a better cost metric
  ("weighted node count") that actually predicts runtime — plain node count badly
  understates the shell explosion this project measured (97% of pops at e≥2).
- **Kucherov, Salikhov, Tsur** — the combinatorial foundation; proves the **connectivity
  property** (the "`pi` must stay contiguous" constraint in `IDEAS_A_E_F.md` §1.5 is a
  theorem, not a heuristic) and p ≥ k+1 (TCS 638:145–158, 2016 / CPM 2014).
- **SeqAn3** — hardcoded optimum schemes for k=0,1,2,3 in
  `include/seqan3/search/detail/search_scheme_precomputed.hpp`; **k=4 is an explicit
  `// TODO … computation has not finished`**, independently confirming k≥4 optima come from
  Renders, not SeqAn.

**Recommended adoption (full tables in §A):**

| k | p | #searches | source | note |
|---|---|---|---|---|
| 3 (L≈30) | 5 | 4 | SeqAn3 `optimum_search_scheme<0,3>` / Renders | best balance |
| 3 | 4 | 3 | Kianfar | fewer searches, heavier tails |
| 4 (L≈46) | 5 | 5 | Renders `multiple_opt/4` | **3 co-optimal variants** |
| 5 | 7 | 6 | hato (`-k 5`) or Gottlieb heuristic | not pre-shipped; generate once offline |
| 6 (L≈91) | 7 | 7 | Renders `multiple_opt/6` | **4 co-optimal variants** |

This is consistent with the `IDEAS_A_E_F.md` estimate (k=4 needs ~5–8 searches ⇒ 2.5–5x):
Renders gives exactly 5 (k=4) and 7 (k=6).

### 2.1 Why Renders beats Kianfar *for this workload*

Kianfar's k=3/k=4 schemes use only 3 searches but each ends at terminal U = k (the full
budget) — few searches, **heavy tails**. Renders uses more searches (5 / 7) in which the
**first-searched part carries L=U=0 or 0/1** and the U=k bound is reached only on the last,
already-narrowed part; the k=6 scheme even caps intermediate U at **3** (`…{0,1,3,3,6,6,6}`),
deferring the heavy budget to the final extension. Since this kernel's cost is the e≥2 shell
(97% of pops), the Renders shape — "never enter the shell early" — is the better match. Use
Kianfar only for k=1/2; prefer Renders/Gottlieb for k≥3.

### 2.2 Edit distance is settled (an open question, now closed)

`IDEAS_A_E_F.md` §1.7 / `PHASE6_SEARCH_SCHEMES.md` flagged "published schemes are Hamming;
edit distance needs boundary slack." The literature is unanimous that this is a non-problem:
Gottlieb — "all our illustrations are for Hamming distance, but the schemes are equally valid
for the edit distance"; **Columba runs edit distance as its default metric using the same
Hamming tables**, tracking indels with a banded DP and ±slack at part boundaries. The slack
**only adds false positives, never false negatives** — exactly this engine's superset
contract (false positives fall through to the CPU reconcile). No "edit-distance-optimal"
tables exist; the community practice is Hamming-optimal scheme + edit-distance slack +
reconcile. **Do the same.**

### 2.3 Tooling for validation and generation

- **Columba `validitychecker/` (Python)** — checks connectivity + monotone L/U + full
  error-configuration coverage for any hardcoded scheme. **Run it on every table before
  flashing to `__constant__` memory** (this is the exhaustive offline covering test
  `IDEAS_A_E_F.md` §1.7.3 asks for).
- **hato** — `./hato solver -k <n>` emits the minU optimum scheme; `-c <idx>` selects among
  co-optimal even-k variants; also an `expNodes` mode to predict expected node count for a
  given (n, m, σ). Requires CPLEX 22.11+ (free academic license); the ILP is small enough
  (k≤6) to port to OR-Tools/GLPK/CBC if CPLEX is unwanted.
- **SeqAn3 `search_scheme_algorithm.hpp`** — a CPU reference executor; diff the GPU hit-set
  against it on a read batch (same tables ⇒ any divergence is an executor bug, not a scheme bug).
- **Search ordering:** SeqAn sorts searches "easy-first" by U-string so the
  cheapest/most-likely-to-reject runs first — port this (it compounds with the per-read
  early-exit on first hit).

### 2.4 GPU novelty — confirmed

Targeted searches for GPU/CUDA + search schemes + bidirectional FM-index return only the CPU
work above. **No published work combines GPU + bidirectional FM-index + optimal search
schemes.** The `PHASE6_SEARCH_SCHEMES.md` publishability claim still stands.

---

## 3. The ~2.4x kernel↔primitive gap: per-lane MLP is the lever

`PROGRESS.md` D6/D7 measured the kernel at 1.7 G-occ4/s vs 4.14 G/s for the same Occ
primitive at the same 8 warps/SM, and named the suspect: "the per-wave serial structure
(pop → probe → syncwarp → prefix-sum → push, MLP=1 per lane) and warp divergence," pending
Nsight Compute. The literature both confirms the diagnosis and supplies the fix.

### 3.1 The diagnosis is memory-level-parallelism starvation

The isolated Occ benchmark reaches 4.14 G/s (and the raw 64 B gather 5.84 G/s = 374 GB/s)
because it feeds the LSU a deep queue of **independent** probes. The kernel does not: each
wave is `pop → one dependent gather → __syncwarp → 5-step shfl prefix-sum → emit`, and a
node's children addresses depend on that node's Occ result, so each lane has **one**
outstanding request (MLP=1). At 8 warps/SM the SM has 256 threads but only ~8 issuing memory
requests at any instant; saturating the memory system needs ~32 outstanding requests/SM. The
3090's random-gather ceiling is 374 GB/s ≈ **40% of the 936 GB/s HBM peak** — the missing
60% is MLP, not bandwidth. (Sources: Volkov, "Better Performance at Lower Occupancy," GTC
2010 / "Understanding Latency Hiding on GPUs," UCB EECS-2016-143; CUDA Best Practices Guide
§11.2 "Hiding Register Dependencies": "with a high degree of exposed ILP it is, in some
cases, possible to fully cover latency with a low occupancy.")

This is the textbook Volkov low-occupancy/latency-bound regime — which is exactly why 8
warps/SM is optimal and raising occupancy hurts (more warps only add L2 pressure on a random
gather without adding useful MLP).

### 3.2 The fix: software-pipelined prefetch / per-lane mini-frontier

Give each lane a **private queue of 2–4 nodes** and overlap their dependent chains: issue
the bucket gather for node *i+1* before consuming node *i*'s result. A node's bucket address
is a pure function of its `(k,l)`, so once the wave reserves its run in the shared stack the
next frontier node's bucket can be prefetched before the current gather retires. (You cannot
prefetch a node's *children* — unknown until Occ returns — but you can prefetch the next
*sibling/frontier* node, which is what breaks the stall.)

The mechanism is available on sm_86 two ways:
- **`prefetch.global.L1/L2 [a]`** (PTX, sm_20+) and the `.level::prefetch_size` load
  qualifier (`L2::64B/128B/256B`, sm_75+) — hand-issue a prefetch for the next node's bucket.
- **Plain early `__ldg`** — hoist the next node's bucket load ahead of the `__syncwarp` and
  prefix-sum so it is in flight during the ALU/sync work (PTX ISA §6.6: "issue the load
  instructions as early as possible, as execution is not blocked until the desired result is
  used"). Verify in SASS that the `LDG` is not sunk to its point of use.

**Direct evidence this works on this exact structure:** QuadRank (Groot Koerkamp, SEA 2026,
arXiv:2602.04103; `github.com/RagnarGrootKoerkamp/quadrank`) started from **bwa-mem's
`bwt.c` rank** — the same 64-byte "4×u64 checkpoint + 128 bp 2-bit" cache-line layout this
engine uses — and measured **"prefetching gives an additional 2x speedup"** by batching
queries and prefetching `queries[i+32]` ahead of `rank(queries[i])`. On CPU the prefetch is
explicit; **on GPU, occupancy is the prefetch buffer**, and where occupancy is capped (here,
by shared memory) the equivalent is per-lane batching/ILP.

**Bit-exactness:** reordering *which* node a lane processes next does not change the visited
node *set*; `has_hit` is order-independent (`IDEAS_A_E_F.md` §0), and the exact ordered
result is regenerated on the CPU reconcile. So per-lane ILP/prefetch is safe for the
detection path. (It would *not* be safe for any emission-order-dependent path — but `has_hit`
isn't.)

**Estimate / risk:** the most likely place to recover a large fraction of the 2.4x;
conservatively target 1.7 → ~2.8–3.4 G-occ4/s. Main risk is **register pressure** from
holding 2–4 nodes' `(k,l,state)` per lane — this project fought hard to reach 0 B local
memory at 79–80 registers (`PROGRESS.md` D5/Phase 6); budget against the 64-reg sweet spot
and watch `ptxas -v`. Prototype behind a macro alongside the existing `STAIR_OK`/`ADD2`
scaffolding.

### 3.3 Read-batching (the structural variant; also the de-divergifier)

The same MLP gain can be obtained structurally: **batch several independent reads per warp /
coarsen so each thread carries probes from multiple reads in flight**, so read A's probe
latency overlaps read B's. This is QuadRank's batching and Chacón et al.'s "cooperative
threads on larger blocks" (below), and it *also* smooths warp divergence near the leaves,
where today `n_active = min(sp,32)` collapses and most lanes issue no gather exactly when the
wave overhead (syncwarp + prefix-sum + `__any_sync` vote) is still paid in full. A per-lane
mini-frontier (§3.2) and read-batching are the two implementations of one idea; both keep
more lanes issuing loads per wave. Bit-exact for the same order-independence reason.

This is the current literature's consistent answer for GPU FM-index speed: **G³SA** (Han,
Kim, Park, Lee, ICS 2025, doi:10.1145/3721145.3729516) names the bottleneck as "irregular
and redundant computation and memory access patterns" and wins by attacking exactly that, not
the rank layout.

---

## 4. k-step FM-index (reduce probe count; bit-exact; memory is the killer)

Precompute the composition `C[c] + Occ(c, ·)` over *k* symbols so one table lookup advances
the backward search by k bases. The result interval is the **identical** `C+Occ` composition,
just tabulated — **bit-exact by construction** (no approximation).

Sources: **COFI** — Langarita et al., "Compressed Sparse FM-Index: Fast Sequence Alignment
Using Large K-Steps," IEEE/ACM TCBB 19(1):355–368, 2022, doi:10.1109/TCBB.2020.3000253
(enables a 15-step human-genome FM-index in <16 GB; ~1.4x over an already-optimised CPU
baseline); the original **n-step** paper — Chacón et al., Procedia Computer Science 18:70–79,
2013, doi:10.1016/j.procs.2013.05.170; hardware analogue **EXMA** (Jiang & Zokaee, HPCA 2021,
arXiv:2101.05314, +4.9x over PIM baselines).

**Honest ceiling ~1.3–1.5x, lower here.** COFI's ~1.4x is on CPU where the workload stays
memory-bound; k-step cuts probe *count* but each probe still misses in DRAM. Two penalties
specific to this engine:

1. **Memory.** This index is `S = T.revcomp(T)`, seq_len ≈ 6.28e9, so a COFI-style table
   roughly *doubles* the 3 Gbp figures. A large-k table (≥16 GB) will not co-reside with the
   3.14 GB index in 24 GB. Realistically limited to **small k (2–4)**.
2. **Approximate backtracking.** k-step helps most on a long *exact-match spine*. bwa-backtrack
   is an *approximate* DFS that branches at mismatches/indels; branched paths are short and
   still pay a probe per symbol. The ~40k probes/read are dominated by combinatorial
   exploration, not a clean spine, so the *effective* probe reduction is below the textbook
   factor — realistically **1.2–1.4x**, and it converts to wall-clock only in proportion to
   how memory-bound the kernel is at the time.

**Recommendation:** prototype **k=2** first (small table, halves the exact-spine probes,
clearly bit-exact) and measure before committing to larger k. It is multiplicative with §3
(MLP) and Idea A (fewer probes × cheaper probes), but it is the weakest of the three and
carries real engineering + memory cost. Note k-step composes awkwardly with the bidirectional
search-scheme engine (the table must support both directions) — sequence it after Idea A.

---

## 5. CompileIQ — the jagged codegen landscape, automated

`PROGRESS.md` recorded maxrregcount 48 → 6,445 r/s, 40 → 5,243, default → 9,934 — a brittle
ptxas scheduling cliff. **NVIDIA CompileIQ** (`github.com/NVIDIA/CompileIQ`,
nvidia.github.io/CompileIQ) is a hyperparameter optimiser for the **undocumented PTXAS
"Advanced Controls"** (per-instruction scheduler knobs); it searches them against your actual
benchmark and emits an Advanced Control File you ship with the kernel. It explicitly "adds
another optimization path … after source-level tuning has plateaued," and ships curated
search spaces pinned by compiler version (`PtxasSearchSpace(version="13.3")`).

This is the honest payoff of "CUDA 13 on sm_86" (§1): a newer ptxas to search. The kernel is
one tight loop — an ideal CompileIQ target — and the objective (reads/s on sub100k) plus the
correctness gate (md5 `eecf35c1`) are already defined in `OPTIMIZATION_IDEAS.md`. Cheap,
automated, reversible; can be run overnight. **Do not** chase `maxrregcount` to raise
occupancy — the kernel is shared-memory-limited at 8 warps/SM and more warps are slower; the
lever is the schedule, not the register count. (The `enable_smem_spilling` pragma — spill to
shared before local — is unlikely to help: local memory is already eliminated and shared is
the scarcest resource.)

---

## 6. Small / cheap experiments

- **`__ldcs` streaming hint on once-touched BWT words** (`ld.global.cs`, evict-first,
  sm_80+). Each BWT scan word and checkpoint line is read exactly once per probe — the
  definition of streaming. Marking those loads `__ldcs` tells L2 not to promote them, so a
  random 64 B line cannot evict the genuinely reusable lines (root/seed buckets, `c_L2`,
  per-read `seq`/`w`/`bid`). One-line change per load; upside 0–5%, risk ~0. A/B it. Note
  `__ldg` (`ld.global.nc`) already routes through the read-only path on Ampere, so measure
  rather than assume. (PTX ISA §9.7.9.1/§9.7.9.2.)
- **`lop3` PTX** in Occ / FM-address arithmetic. The N-Queens solver (arXiv:2511.12009)
  fused 3-input bitwise ops with `lop3` for +7% over compiler output. If `d_occ_aux4_popc` or
  the interval arithmetic has 3-input AND/OR/NOT combinations, `lop3` fuses them to one
  instruction. Low priority; check SASS.
- **Bank-conflict audit.** The shared stack is SoA per-warp; the pop phase
  (`stack[sp-1-lane]`) is already conflict-free, but the post-prefix-sum push could collide.
  Check Nsight `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`; only if non-trivial, apply
  XOR padding. Likely ~0 given 81% lane utilisation and the MLP diagnosis.

---

## 7. Multi-GPU (when the A4000 arrives)

The A4000 is **sm_86** (same arch as the 3090), 48 SM / 448 GB/s vs 82 SM / 936 GB/s — roughly
1.7:1 compute, 2.1:1 bandwidth. The Phase-5 ready-queue already has the right shape. The
literature (and the N-Queens paper's multi-GPU section, which saw >100% per-GPU time variance
under *uniform* static partitioning) says: use **batch-level dynamic distribution** — a
host-side atomic counter feeds fixed read batches (e.g. 4096 reads) to whichever GPU finishes
first; the 3090 naturally pulls ~1.7x more. Per-GPU persistent kernels share identical code;
the existing ordered finisher / CPU reconcile is the synchronization point. Expected ~90% of
ideal; ~1.5x combined. Heterogeneous-aware refinements (Blackwell **Cluster Launch Control**
hardware work-stealing; CUDA **Green Contexts**) are **sm_100+ / sm_90+ and unavailable
here** — future-proofing notes only.

---

## 8. Rejected on evidence (do not re-explore)

| idea | why dead here | source |
|---|---|---|
| L2 persistence / `cudaAccessPolicyWindow` / eviction-priority hints | BWT 3.14 GB ≫ 6 MB L2; gathers random with no temporal reuse — nothing to pin. Confirms the project's own rejection. | PTX ISA §9.7.9.2/17/19; Best Practices §10.2.2 |
| `cp.async` / `cuda::pipeline` / `memcpy_async` for the gather | built for large predictable tiled loads; the gather is a single dependent 64 B line consumed immediately — routing through shared adds a round-trip + `wait_group` without removing the dependency. `cp.async.bulk`/TMA is sm_90+ anyway. | PTX ISA §9.7.9.26; Best Practices §10.2.3.4 |
| warp specialization (producer/consumer) | compute per probe is a few `__popc`/ALU + a 5-step scan, dwarfed by the gather — no compute to hide *behind* memory; the useful overlap is memory-with-memory (= §3), and specialization would shrink the thin 8-warp occupancy. | — |
| raise occupancy | shared-memory-limited at 8 warps/SM; 24 warps/SM measured slower; more warps add L2 pressure on a random gather without adding useful MLP. | `PROGRESS.md` Phase 6 |
| 32 B / two-level Occ layout (Idea B) | 1.16x at the shipped 8 warps/SM (request-rate-limited, not byte-limited); kernel not memory-bound. Already downgraded in D7. | `PROGRESS.md` D7 |
| compressed layouts (RLFM, prefix-free, wavelet) | byte savings hit the same 1.16x request-rate ceiling; RLFM rank is branchy (bad for a divergence-sensitive kernel); bwa layout fixed for compatibility. The project already ran this experiment (64→32 B = 1.16x). | QuadRank §2 survey |
| popcount / broadword / `__byte_perm` | `__popc` is single-cycle hardware on sm_86 and is already shipped (+18%); primitive at 4.14 G/s ⇒ compute is not the constraint. No CUDA analogue of AVX2 nibble-shuffle beats hardware `__popc`. | QuadRank §4 |
| caching / tiling / software bucket cache | 0.04% within-read bucket reuse over ~49 M buckets; below miss-penalty breakeven. The 25 MB-flat working-set sweep is the empirical proof. Distinguish *latency-hiding* prefetch (§3, yes) from *reuse* caching (no). | `PROGRESS.md` D1 |
| checkpoint re-spacing | buys compute (not the bottleneck) or index size; cannot shrink 3.14 GB → <6 MB L2. | QuadRank overhead ladder |
| dynamic parallelism (CDP1/CDP2) | per-child-grid launch overhead catastrophic at millions of reads; `cudadevrt` linkage penalises all kernels; no shared-memory sharing across grids; CDP2 removed device-side sync. Persistent threads + atomic pool already achieve it at zero overhead. | CUDA Programming Guide §4.18 |
| cluster launch control / Blackwell work-stealing / Green Contexts | sm_100+ / sm_90+ only. | CUDA Programming Guide §4.12 |
| N-Queens bank-conflict layout (128 B stride) | solves one-thread-per-stack; this engine is warp-cooperative (32 lanes hit 32 *different* entries), so the trick is irrelevant. Citation is prior-art only. | arXiv:2511.12009 |
| pure-BFS / global-frontier engine | frontier width varies 6 orders of magnitude; a 1.27 M-node read = 25 MB global frontier compacted every level vs a 10 KB shared stack. DFS wins; the existing wave/drain hybrid is the right structure. | GPU graph-traversal literature |

### Corrections to earlier documents

- The "Faster and Cheaper: Pushing the Sequence Alignment Throughput" paper (`REFERENCES.md`)
  is **PPoPP 2026** (not ICS), doi:10.1145/3774934.3786421, and is about **multi-stage
  seeding + intra-query parallel seed-extension on CPU** — *not* Occ checkpoint spacing. Its
  relevance here is indirect ("100% BWA-MEM-identical output" reconfirms bit-exact parity is
  achievable; "eliminate redundancy" echoes §3.3).
- **G³SA** and the "Han, Kim, Park, Lee" entry in `REFERENCES.md` are the **same paper**
  (ICS 2025, doi:10.1145/3721145.3729516).
- A full **BWA-MEM-on-GPU** now exists (not just BarraCUDA/NVBIO): Pham, Tu, Lv,
  "Accelerating BWA-MEM Read Mapping on GPUs," ICS 2023, doi:10.1145/3577193.3593703,
  `github.com/minhhpham/bwa` (up to 3.2x over BWA-MEM2 on an A40). It is seed-and-extend, so
  its *algorithm* is still wrong for `-l 1024` aDNA, but its GPU plumbing is a useful
  reference for the seeding/extension stages.

---

## 9. Recommended sequencing (updated)

1. **Idea A step 1 — measure the `bwt_cal_width` bound loss** for interior-anchored searches
   (`IDEAS_A_E_F.md` §1.7.1). Still the biggest unknown; cheap; do before building the engine.
2. **Idea A step 2 — transcribe the Renders/SeqAn tables** (§A), run Columba's
   `validitychecker` on each, and diff the GPU executor's hit-set against SeqAn3's CPU
   executor. This is now a build task, not research.
3. **Idea A step 3 — build the per-search engine** on the Phase-6 skeleton; validate md5s.
4. **In parallel, the MLP lever (§3):** prototype per-lane 2–4-node prefetch / read-batching
   behind a macro; it is orthogonal to the search algorithm and attacks the diagnosed 2.4x
   gap directly. Watch `ptxas -v`.
5. **Overnight, CompileIQ (§5):** zero code change; clean objective + correctness gate.
6. **E (dedup) and F (CPU worker):** as in `IDEAS_A_E_F.md` — real, small, do when convenient.
7. **k-step k=2 (§4):** only after Idea A, and only if probe count (not control flow) is
   still the limiter.
8. **Multi-GPU batch queue (§7):** when the A4000 is installed.

Independent of all of the above: `sudo pacman -S nsight-compute` would still settle whether
the remaining gap after §3 is per-wave pipeline bubbles (`smsp__inst_executed_pipe_lsu`,
`memory_throughput`) — but §3 gives a concrete fix to try *without* waiting on it.

---

## A. Appendix — hardcode-ready search-scheme tables

Verbatim from the cited repositories. Format `{pi} {L} {U}`, **0-based parts**, contiguous
`pi`, cumulative error bounds. **Verify the covering property offline (Columba
`validitychecker`) before use** — these are literature tables, not yet validated on this
engine. For edit distance, execute with ±1 boundary slack and reconcile false positives on
the CPU (lossless superset; §2.2).

### A.1 Kianfar (Columba `search_schemes/kianfar/`) — use for k=1,2

```
k=1  (2 searches, p=2)
{0,1} {0,0} {0,1}
{1,0} {0,0} {0,1}

k=2  (3 searches, p=3)
{0,1,2} {0,0,2} {0,1,2}
{2,1,0} {0,0,0} {0,2,2}
{1,2,0} {0,1,1} {0,1,2}

k=3  (3 searches, p=4)        # heavy tails; prefer SeqAn/Renders p=5
{0,1,2,3} {0,0,0,3} {0,2,3,3}
{1,2,3,0} {0,0,0,0} {1,2,3,3}
{2,3,1,0} {0,0,2,2} {0,0,3,3}

k=4  (3 searches, p=5)        # heavy tails; prefer Renders
{0,1,2,3,4} {0,0,0,0,4} {0,3,3,4,4}
{1,2,3,4,0} {0,0,0,0,0} {2,2,3,3,4}
{4,3,2,1,0} {0,0,0,3,3} {0,0,4,4,4}
```

### A.2 Renders et al. (Columba `search_schemes/multiple_opt/`) — use for k=4,6

```
k=2  — 2 co-optimal variants, 3 searches, p=3
 scheme1:                       scheme2 (mirror):
 {0,1,2} {0,1,1} {0,2,2}        {2,1,0} {0,1,1} {0,2,2}
 {1,0,2} {0,0,0} {0,1,2}        {1,2,0} {0,0,0} {0,1,2}
 {2,1,0} {0,0,2} {0,1,2}        {0,1,2} {0,0,2} {0,1,2}

k=4  — 3 co-optimal variants, 5 searches, p=5
 scheme1:
 {0,1,2,3,4} {0,0,2,2,2} {0,2,2,4,4}
 {1,2,0,3,4} {0,0,0,0,0} {0,1,2,4,4}
 {2,1,0,3,4} {0,1,1,1,1} {0,1,2,4,4}
 {3,4,2,1,0} {0,0,0,0,3} {0,1,4,4,4}
 {4,3,2,1,0} {0,1,1,1,4} {0,1,4,4,4}
 scheme2:
 {0,1,2,3,4} {0,1,1,1,4} {0,1,4,4,4}
 {1,0,2,3,4} {0,0,0,0,3} {0,1,4,4,4}
 {2,3,4,1,0} {0,1,1,1,1} {0,2,2,4,4}
 {3,2,4,1,0} {0,0,0,0,0} {0,1,2,4,4}
 {4,3,2,1,0} {0,0,2,2,2} {0,1,2,4,4}
 scheme3 = reverse-mirror of scheme2 (in repo)

k=6  — 4 co-optimal variants, 7 searches, p=7
 scheme1:
 {0,1,2,3,4,5,6} {0,0,2,2,2,2,6} {0,2,2,6,6,6,6}
 {1,2,0,3,4,5,6} {0,1,1,1,1,1,5} {0,1,2,6,6,6,6}
 {2,1,0,3,4,5,6} {0,0,0,0,0,0,4} {0,1,2,6,6,6,6}
 {3,4,5,6,2,1,0} {0,0,0,0,0,0,0} {0,1,3,3,6,6,6}
 {4,3,5,6,2,1,0} {0,1,1,1,1,1,1} {0,1,3,3,6,6,6}
 {5,6,4,3,2,1,0} {0,0,0,2,2,2,2} {0,1,3,3,6,6,6}
 {6,5,4,3,2,1,0} {0,1,1,3,3,3,3} {0,1,3,3,6,6,6}
 (scheme2/3/4: co-optimal mirrors/permutations, all in repo)
```

### A.3 k=3 (SeqAn3 `optimum_search_scheme<0,3>`, converted to 0-based) — 4 searches, p=5

```
{4,3,2,1,0} {0,0,0,0,0} {0,0,3,3,3}
{2,3,4,1,0} {0,0,1,1,1} {0,1,1,2,3}
{1,2,3,4,0} {0,0,0,2,2} {0,1,2,2,3}
{0,1,2,3,4} {0,0,0,0,3} {0,2,2,3,3}
```

Cross-check (Gottlieb's independently derived k=3, p=5):
`s0=(01234,00003,02233) s1=(12340,00022,01223) s2=(23410,00111,01123) s3=(34210,00000,00333)`.
SeqAn3 also ships `<1,3>/<2,3>/<3,3>` guaranteed-min-error variants if the budget is ever
split into a guaranteed-min-error first search.

### A.4 k=5 — generate offline

Not pre-shipped (Columba ships only even k for the optimum family). Either
`./hato solver -k 5` (CPLEX) or Gottlieb's closed-form heuristic (k+1 = 6 searches, p = k+2
= 7, verified valid+complete to k=15). Validate with Columba's checker, then hardcode. Low
priority: the GPU sees k=5 only in a narrow length band near the 64 bp short/long split.

---

## B. Citations

Search schemes: Kianfar et al., arXiv:1711.02035 (RECOMB-Seq 2018) · Renders et al.,
J. Comput. Biol. 31(10):975–989 (2024), doi:10.1089/cmb.2024.0664 / RECOMB 2024,
doi:10.1007/978-1-0716-3989-4_11 · Gottlieb & Reinert, NAR Genom. Bioinform. 7(1):lqaf025
(2025), PMC11915513 · Depuydt et al., "Search Schemes: An Overview," OASIcs vol. 131
(Manzini), 9:1–16, doi:10.4230/OASIcs.Manzini.9 · Kucherov, Salikhov, Tsur, TCS 638:145–158
(2016). Code: github.com/biointec/columba (schemes + checker), github.com/biointec/hato
(MILP), github.com/seqan/seqan3 (executor + k≤3 optima).

Rank / FM-index: QuadRank — Groot Koerkamp, SEA 2026, arXiv:2602.04103,
github.com/RagnarGrootKoerkamp/quadrank · COFI — Langarita et al., IEEE/ACM TCBB 19(1), 2022,
doi:10.1109/TCBB.2020.3000253 · n-step — Chacón et al., Procedia CS 18:70–79, 2013,
doi:10.1016/j.procs.2013.05.170 · Boosting the FM-index on the GPU — Chacón et al., IEEE/ACM
TCBB 12(5):1048–1059, 2015, doi:10.1109/TCBB.2014.2377716 (PMID 26451818) · EXMA — Jiang &
Zokaee, HPCA 2021, arXiv:2101.05314.

GPU alignment: G³SA — Han, Kim, Park, Lee, ICS 2025, doi:10.1145/3721145.3729516 ·
Accelerating BWA-MEM on GPUs — Pham, Tu, Lv, ICS 2023, doi:10.1145/3577193.3593703,
github.com/minhhpham/bwa · FastAlign — Zhang et al., PPoPP 2026, doi:10.1145/3774934.3786421,
github.com/zzhofict/BWA-FastAlign · WFA-GPU — Aguado-Puig et al., Bioinformatics 2023,
doi:10.1093/bioinformatics/btad701 · AGAThA — Park et al., PPoPP '24,
doi:10.1145/3627535.3638474 · BWA-MEM-SCALE — Kim et al., ICPP '22, doi:10.1145/3545008.3545033.

CUDA / GPU tree search: CUDA 13.3 release notes,
docs.nvidia.com/cuda/cuda-toolkit-release-notes · PTX ISA 9.3,
docs.nvidia.com/cuda/parallel-thread-execution · CUDA Best Practices Guide §10.2/§11.2/§12.2/§13,
docs.nvidia.com/cuda/cuda-c-best-practices-guide · CompileIQ, github.com/NVIDIA/CompileIQ ·
Volkov, "Better Performance at Lower Occupancy," GTC 2010 / "Understanding Latency Hiding on
GPUs," UCB EECS-2016-143 · N-Queens GPU DFS — Yao & Li, arXiv:2511.12009, github.com/ygch/n_queens ·
Component-aware vertex cover on GPU — Amro et al., IEEE TPDS 37(2), 2026, arXiv:2512.18334,
github.com/HusseinAmro/component-aware-vertex-cover-gpu · CUDA Dynamic Parallelism §4.18 and
Cluster Launch Control §4.12, CUDA Programming Guide.
