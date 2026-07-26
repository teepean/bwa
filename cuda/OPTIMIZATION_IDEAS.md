# Optimization roadmap (user ideas + critical assessment)

Current standing: `cuda/dfstest.cu`, 9934 reads/s on sub100k = ~2.3x the 16-core CPU, bit-exact.
Bottleneck = global-memory DFS stack (raising occupancy hurt; 2occ4 fast path didn't help).

## 1. Warp-level parallel-branch DFS + shared-memory stack  — ADOPTED (next move)
One warp per read; the per-warp stack lives in **shared memory** (spill to global only on
overflow); the warp expands multiple frontier nodes per iteration (32 lanes) instead of one
thread serializing one path. Directly attacks the diagnosed bottleneck: stack moves off global
memory, and stack ops become warp-aggregated shared-memory ops.
- Occupancy note: one-read-per-warp gives ~2.3k reads in flight (vs ~73k now) but the SAME
  ~28 warps/SM, so latency hiding is preserved while each warp keeps ~32 FM-index probes in
  flight. Lane utilization is fine here because the 99.3% zero-hit reads still have ~40k nodes
  to spread across lanes (they are deep, not shallow).
- Must stay bit-exact: has_hit detection is order-independent, so any traversal covering the same
  bounded node set (and early-exiting on any hit) is valid. Keep the work-budget -> CPU reconcile.
- Combines user ideas #1 (warp branching) and #2 (register + SMem stack window).

## 2. Register + shared-memory stack window — ADOPTED (part of #1)
Top of stack in registers (already done via in-register-continue); next slice in per-warp shared
memory; global only on SMem overflow. 16 entries x 32 lanes x 20 B ~= 10 KB/warp — fits.

## 3. "Zero-hit" exact-seed fast path — REJECTED (breaks correctness for -l 1024)
Proposal: if the first l (e.g. 32) bases don't match exactly, mark the read unmappable and skip.
**This is incorrect for our use case.** With `-l 1024` seeding is DISABLED precisely because
ancient-DNA reads carry mismatches/deamination anywhere, including the first 32 bp. A read can map
with several mismatches in that region; an exact-seed filter would wrongly skip it -> false
negatives -> non-bit-exact `.sai`. (This is the whole reason aDNA pipelines use -l 1024.)
- A *conservative* prefilter that only skips reads PROVABLY unmappable within max_diff (e.g. a
  q-gram/minimizer count lower bound) could be bit-exact-safe and is worth exploring later, but the
  exact-seed version is not safe. Deferred, low priority.

## 4. Vectorized Occ loads — PARTIAL / minor
The BWT bucket is already a 64 B cache line; `__occ_aux4` already unpacks 4 nt per 32-bit word via
cnt_table (so the "unpack 4 per load" is done). Possible minor win: fetch the 4xuint64 checkpoint
as two `int4`/`uint4` `__ldg` instead of 4 scalar loads. Low priority; do after #1.

## 5. Pipelining: overlap CPU reconcile with GPU via streams — ADOPTED (for the full-file engine)
Currently the harness is synchronous. The production full-file path must: stream read batches,
run the DFS kernel on stream A while a host thread pool runs `bwt_match_gap` on the previous
batch's flagged reads, double-buffering. The 1-thread 2.37 s reconcile becomes ~0.15 s on 16
threads and overlaps the kernel. Do when building the end-to-end streaming engine.

## 6. #pragma unroll / template on read length — LOW priority
The DFS is a tree, not a length-bounded loop, so templating read length gives little. The bounded
inner loops (within-bucket occ, 0..8) are already unrollable. Revisit only if profiling shows it.

## Round 2 (post-ceiling) ideas + verdicts — tested by the occ4 locality sweep
The occ4 memory-hierarchy sweep (FMTEST_KRANGE) settles these empirically:
L1/L2-resident occ4 = ~9.9 G/s; HBM (full 3.14 GB) = ~2.3 G/s -> a 4.3x locality ceiling EXISTS in
cache. BUT the warp2 kernel already runs at ~1.8 G occ4/s = the HBM-random rate (not the L2 rate),
because the FM-index DFS is pure scatter: ~40k probes/read over ~49M 64-byte buckets => ~0.04%
within-read block reuse (birthday bound). So:
- #1 SMEM BWT block cache: REJECTED. Per-warp cache captures only within-warp reuse, of which there
  is ~none; cross-read hot-root blocks are <0.1% of probes and already L2-resident. The 4.3x ceiling
  is unreachable for this access pattern.
- #2 warp address-sort/dedup: REJECTED. 32 probes over 49M buckets ~never share a DRAM row; sort
  cost > benefit.
- #3 bit-matrix popc (drop cnt_table): REJECTED for speed. Memory-bound, not compute-bound
  (cnt_table is constant-cache); same bytes/probe -> no bandwidth change.
- #4 multi-GPU round-robin: REAL ~2x (doubles HBM bandwidth); embarrassingly parallel across the
  0x40000 chunks. Needs a 2nd GPU (not present). Worth scaffolding.
- #5 CPU/GPU double-buffer overlap: REAL ~10% wall-clock (hide ~18s preprocess+reconcile behind the
  ~154s GPU). The only single-GPU win left. Bit-exact-safe if samse stays a single ordered consumer.
CONCLUSION: kernel is at the HBM random-access BANDWIDTH wall. Only more bandwidth (multi-GPU) or
fewer probes (k-step FM-index; marginal for a branching search, risks bit-exactness) move it.

## Order of execution
1) #1+#2 warp-cooperative DFS with shared-memory stack (biggest lever).
2) #5 streaming + multithreaded overlapped reconcile (end-to-end wall-clock).
3) #4 vectorized checkpoint load; then re-profile.
Re-check CUDA docs + NVIDIA forums at each step (cuda/REFERENCES.md).

---

## Round 3 — Phase 6 ideas (post-ceiling, breaking the occ4 wall)

> **Detailed design for the three unshipped ideas (A, E, F) is in `cuda/IDEAS_A_E_F.md`** --
> what each is, why it is bit-exact, everything measured, the implementation plan, and the open
> problems (notably the loss of the `bwt_cal_width` bound for interior-anchored searches, and the
> fact that a monotone staircase is NOT a covering scheme).

Full analysis in cuda/PHASE6_SEARCH_SCHEMES.md. Summary here.

### Idea A — Bidirectional search schemes (3-13x, the exponent changer) — RESEARCH TRACK

The GPU's only obligation is has_hit_gpu(r) >= has_hit_bwa(r) (no false negatives).
bwa's index S = T.revcomp(T) with revcomp(S)=S is already bidirectional (bwt.c:262
bwt_extend). d_bwt_2occ4 returns all 4 counts — the exact primitive needed. No second
index, no extra memory.

Search schemes partition the read into p parts with staircase error budgets, collapsing
the combinatorial shell (41k -> 1-13k node-pops). 2-part pigeonhole: ~3.2x. 3-part
staircase: ~13x. Kianfar/Pockrandt report 35x for 101bp/k=2; our k=3-4 at L=30-64
gives 3-13x credibly.

Node format grows from 20 B to 24 B (bwtintv_t triple). CAP_SM=512 -> 12 KB/warp,
still 8 warps/SM on sm_86.

Caveats: (1) published schemes are Hamming; edit distance needs boundary slack (cheap,
superset only); (2) max_diff varies per read (3/4/6); (3) false-positive rate must stay
near 0.531% or reconcile becomes the bottleneck.

No published work combines GPU + bidirectional FM-index + optimal search schemes.
Publishable result.

### Idea B — Two-level Occ layout — DOWNGRADED to ~1.16x (D7), DO NOT BUILD YET

Current 64-byte bucket = [4xu64 checkpoints (32B)][8xu32 BWT (32B)]. Half is
checkpoint overhead. Repack: superblock (65536 bases, 4xu64 absolute, 3.07 MB,
L2-resident) + block (32B = 4xu16 relative + 24B seq = 96 bases). One 32-byte
sector per probe instead of 64. Index 3.14 GB -> 2.09 GB. With 3-count refinement:
1.93 GB. Bit-exact by construction. Pair with __ldcs on block loads.

The earlier "2-bit compressed BWT" idea missed this: the BWT is ALREADY 2-bit packed.
The win is deleting absolute checkpoints from the hot path.

**VERDICT (D7): the 1.89x justification was itself a high-occupancy artefact.** Re-measured
best-of-3 vs resident warps: 32 B beats 64 B by 1.93x at 32 warps/SM but only **1.16x at the
shipped 8 warps/SM**. At low occupancy the memory system is limited by request rate/latency,
not bytes. 1.16x does not pay for re-laying out a 3.14 GB index, and the kernel is not
memory-bound anyway (1.7 vs 4.14 G-occ4/s at matched occupancy). Revisit only if the remaining
~2.4x per-wave gap is closed and the kernel becomes genuinely bandwidth-bound.

### Idea C — cnt_table -> bit-sliced __popc — DONE, +18%, shipped as default

d_occ_aux4 (fm_device.cuh:58) does 4 divergent c_cnt_table constant-memory lookups
per u32 word. 32 lanes x different addresses = serialized constant-cache reads, up to
32 per probe. Replace with 4 __popc + ~6 ALU ops, zero memory. Win is from eliminating
constant-cache serialization stalls, orthogonal to bandwidth.

**VERDICT: DONE (+18%, bit-exact, fmtest-validated, default; `-DFM_OCC_CNTTABLE` reverts).**
Mechanism confirmed as constant-cache/instruction cost, NOT register pressure: `ptxas -v` gives
popc = 80 registers vs cnt_table = 79, so popc wins *despite* one extra register. Isolated
occbench at matched 8 warps/SM: 4.43 vs 4.14 G/s = +7%; the full kernel sees +18% because it
also carries `c_L2` and 536 B of params in cmem[0].
Note the "already fully hidden" reading of the fmtest number was true only at HIGH occupancy.

### Idea D — Proactive spill — REJECTED on measurement (D4)

dfs_engine.cuh:118: spill triggers at room < 9, restoring only CHUNK=128. The warp
oscillates at <=44% lane occupancy near capacity. Fix: trigger at room < 288 (32*9).
Keeps n_active=32 for more waves. Cost: more frequent shared<->global copies (not
HBM-scatter).

**VERDICT: REJECTED.** The premise is false. Measured mean active lanes = **25.97/32 (81%)** and
spills = 5057 total = **0.000/wave**: with CAP_SM=512 the frontier never approaches capacity, so
the `room/9` throttle never engages. Nothing to fix.

### Idea E — Read-level deduplication (1.2-1.5x, trivially bit-exact) — INCREMENTAL

bwa aln ignores base qualities. Identical sequences -> identical DFS trees -> identical
has_hit. Run DFS once per distinct (len, seq), broadcast has_hit. aDNA duplicates
heavily; bushy low-complexity reads dominate duplication -> superlinear saving.

Measure: zcat reads.fq.gz | awk 'NR%4==2' | sort | uniq -c | sort -rn | head

### Idea F — CPU as 17th worker (~+15%, free, bit-exact) — INCREMENTAL

CPU does 4,261 r/s on 16 threads; GPU now does ~40k (post-Phase-6). CPU is idle ~85% of wall
time (ratio is now ~10:1, so this is worth ~+9%, not +15%).
Feed it ~15% of reads through the same chunk queue. Pipeline already supports this
(Phase 5 ready-queue). Bit-exact: CPU runs the reference implementation.

### CompileIQ — compiler flag HPO (1.0-1.15x, zero code changes) — OVERNIGHT

maxrregcount 48->6445, 40->5243, default 69->9934 r/s: jagged compiler landscape.
NVIDIA CompileIQ (https://github.com/NVIDIA/CompileIQ) searches this automatically.
Clean scalar objective (reads/s on sub100k) + correctness gate (md5 eecf35c1).
CUDA 13.3 -> PtxasSearchSpace(version="13.3").

### Execution order — UPDATED after Phase 6 (see PROGRESS.md D1-D7)

Diagnostics: **DONE** (D1-D7). Results overturned the Round-2 "HBM wall" framing entirely.
Nsight Compute is the one diagnostic still outstanding (not installed:
`sudo pacman -S nsight-compute`).

Status:
- **C — DONE** (+18%, shipped).
- **D — REJECTED** (D4: 81% lane utilisation, ~0 spills).
- **B — DOWNGRADED** (D7: 1.16x at the shipped occupancy; shelved).
- **The real Phase-6 win was not on this list at all**: 320 B/thread of address-taken LOCAL
  memory (child buffers + `cnt[4]` arrays). Two-pass child generation + scalar-reference occ4
  removed it. Combined with C: **1.82x, bit-exact.**

### Idea G — longest-first work-pool scheduling — DONE, up to 1.59x (Phase 9)
Not on the original list; found by Nsight Compute (SM-active/elapsed = 64%, i.e. 36% straggler
tail). Tree size is exponential in `max_diff`, a step function of read length, so hand out the
longest reads first (LPT). ~10 lines: an `order[]` permutation in the work pool. Bit-exact --
scheduling order only decides which warp takes which read. `GPUALN_NOORDER=1` to A/B.
Gain scales with read-length VARIANCE: 1.50-1.59x on 30-91 bp, 1.07x on 30-63 bp, 1.00x on
30-34 bp. Never negative, so on by default.

Remaining, in order:
1. ~~**MLP per lane**~~ — **BUILT AND REJECTED** (Phase 9). 0.90-0.95x. The frontier only holds
   ~28 nodes (branching factor ~1, so pops == pushes and it self-limits), so a second slot per lane
   is empty and only costs registers. Raising in-flight probes needs a wider frontier = >1 read per
   warp = 2 stacks/warp, which costs either occupancy or spills. Direction closed. Original
   reasoning kept below for the record.
   **MLP per lane** (Phase 9 diagnosis). ncu: 74.7% of cycles have NO eligible warp and 49.1% of
   warp-cycles stall on `long_scoreboard` (memory dependency) with only 1.96 warps/scheduler.
   Occupancy cannot fix it -- re-swept post-Phase-6, 8 warps/SM is still optimal and more warps is
   worse (matches D6: the memory system peaks at 8 warps/SM). The fix is more independent probes
   per LANE: pop 2 nodes/lane/wave and issue both `2occ4` calls before consuming either.
   Register headroom is free here (79 used; shared memory caps us at 256 threads/SM, so up to 256
   registers/thread cost nothing). Secondary: divergence, 10.8/32 active threads (ncu est. 11%).
2. E (read dedup) — untested, still valid, cheap to measure.
3. F (CPU as extra worker) — untested, now ~+9%.
4. CompileIQ — overnight, zero code change.
5. B — only if (1) makes the kernel bandwidth-bound.

Parallel research track (the exponent changer):
A (start with 2-part pigeonhole, validate on sub2k/sub100k). D3 confirms its premise exactly:
predicted 41,000 pops/read vs 40,924 measured, e>=2 = 97% of pops. Estimated 2.5-4x.

Overnight: CompileIQ (no code changes)

Combined realistic target: 15-40x over current 22k r/s on a single RTX 3090.
