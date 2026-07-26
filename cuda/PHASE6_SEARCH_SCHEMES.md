# Phase 6 — Search schemes + representation: breaking the occ4 ceiling

> ## SUPERSEDED IN PART — read this first (2026-07-26)
>
> Phase 6 was **executed**. The diagnostics in this document have been run and several of its
> premises turned out to be wrong. Authoritative results: `cuda/PROGRESS.md` sections **D1-D7**.
>
> | this document says | actual result |
> |---|---|
> | "kernel is at the HBM bandwidth wall (~2.28 G occ4/s)" — framing used throughout | **WRONG (D6).** That figure was measured at HIGH occupancy. At the shipped 8 warps/SM the hardware gives 5.84 G probes/s = 374 GB/s. The kernel was never bandwidth-bound. |
> | §3 Idea B: two-level 32 B Occ layout, 1.5-2x | **DOWNGRADED to 1.16x (D7).** The 1.89x was itself a high-occupancy artefact. Shelved — do not build. |
> | §4 Idea C: popc, win from constant-cache serialization | **DONE, +18%, shipped as default.** Mechanism is constant-cache/instruction cost but NOT register pressure (popc = 80 regs vs 79). |
> | §5 Idea D: lane-utilisation collapse, proactive spill | **REJECTED (D4).** Measured 25.97/32 lanes active (81%), spills 0.000/wave. The collapse does not happen. |
> | §9 "the three diagnostics (run first)" | **Already run** (D1-D7). Only Nsight Compute is outstanding (not installed). |
> | §11 "15-40x over current 22k reads/s" | **Stale baseline.** The kernel is now 1.82x faster; remaining headroom is ~2.4x (per-wave structure) then 2.5-4x (search schemes). |
>
> **The actual Phase-6 win was not in this document at all**: 320 B/thread of address-taken
> LOCAL memory (`cck[9]/ccl[9]/ccn[9]` child buffers + every `uint64_t cnt[4]` passed by
> pointer). Two-pass child generation + scalar-reference occ4 + popc = **1.82x, bit-exact**,
> validated byte-identical on 3.5 M and 671 k read sets.
>
> **What survives unchanged:** §0 (the correctness reframing), §2 (Idea A and the bidirectional
> -index enabler — D3 confirms its cost model exactly: 41,000 pops/read predicted vs 40,924
> measured, `e>=2` = 97% of pops), §6 (Idea E), §7 (Idea F). Idea A remains the headline lever.

Status: EXECUTED — see PROGRESS.md D1-D7 for measured outcomes.
Predecessor: Phase 5 (multi-GPU pipeline, ~22k reads/s single-RTX-3090, bit-exact).

The Phase 4/5 conclusion was that the single-GPU kernel is at the HBM random-access
bandwidth wall (~2.28 G occ4/s, ~22k reads/s). Three independent attempts (pigeonhole
prefilter, vectorized loads, L2 persistence) confirmed this. **All of that is now known to be
an artefact of high-occupancy benchmarking (D6).** The only single-GPU paths
beyond 22k are: (1) fewer probes per read (algorithmic), (2) fewer bytes per probe
(representation). This document analyses both, plus five low-risk incremental levers.

---

## 0. The correctness reframing

The GPU kernel's only obligation is a **conservative superset predicate**:

    has_hit_gpu(r) >= has_hit_bwa(r)    (pointwise, no false negatives)

False positives cost CPU reconcile time, not correctness. Every `has_hit=1` read is
re-run by the exact CPU `bwt_match_gap`:

    aln_gpu.cu:258   if (c->has_hit[i]) idx.push_back(i);
    aln_gpu.cu:280   aln[i] = bwt_match_gap(bwt, len, ..., &na, st);

The GPU is free to run a completely different search algorithm — bidirectional,
search-scheme-based, anything — as long as it never misses a read that bwa would map.

### Why bwa is strictly more restrictive than edit-distance

`gap_init_opt()` (bwtaln.c:32-33):

    o->indel_end_skip = 5; o->max_del_occ = 10; o->max_entries = 2000000;
    o->mode = BWA_MODE_GAPE | BWA_MODE_COMPREAD;

With `BWA_MODE_GAPE` set, the remaining-error budget at bwtgap.c:200-201 is:

    m = max_diff - (e.n_mm + e.n_gapo);
    if (opt->mode & BWA_MODE_GAPE) m -= e.n_gape;

so `m = max_diff - (mm + gapo + gape)` = true edit-distance budget. Additional
restrictions that make bwa a SUBSET of "exists substring within edit distance <= max_diff":

- `width[i-1].bid` prune (bwtgap.c:208): the width-array lower bound on remaining errors
- `indel_end_skip=5` (bwtgap.c:164 in the DFS engine): no indels within 5 bp of read ends
- `max_del_occ=10` (dfs_engine.cuh:174): long deletions pruned if SA interval > 10
- `max_entries=2M` (bwtgap.c:193): the frontier cap aborts pathological reads

Therefore: **bwa maps read r => exists genome substring within edit distance <= max_diff
of r.** The contrapositive gives the GPU's correctness obligation: if no such substring
exists, the GPU must report has_hit=0. Any search that conservatively covers all
edit-distance-<=-max_diff occurrences satisfies this.

---

## 1. Where the work actually is

Node-pop model: N(j) ~ [sum_{e<=E(j)} C(j,e) * 3^e] * min(1, 6.28e9 / 4^j),
with E(j) = max_diff - w_bid[len-1-j]. For L=46, max_diff=4:

    depth j:   13     14      15      16      17     18    20
    nodes:    740   10,700  13,300  11,700   3,500  1,050   80

Total ~ 41,000 node-pops/read, matching the measured ~41k. The mass is a thin shell
at depth 14-18, dominated by the e=3 term C(j,3)*27.

Consequence: k-mer existence bitmaps, ftab/jump tables, singleton-interval collapse,
and L2-resident prefix tables all target depths where there is no work. The cost is
combinatorial (3^e), not genomic. **The only algorithmic lever is reducing the exponent**,
which is exactly what search schemes do.

This can be confirmed with the existing DFS_INSTRUMENT histogram:

    Build:  make bwa-aln-gpu  (add -DDFS_INSTRUMENT to NVCCFLAGS)
    Run:    DFS_INSTR_MOD=1 GPUALN_HISTO=1 ./bwa-aln-gpu -t 16 hs37d5 sub100k.fq > /dev/null

The [instr] output gives the full (depth, errors-used) profile. Set DFS_INSTR_MOD=1
for all reads (slower but exact counts); the default 128 samples 1-in-128.

---

## 2. Idea A — Bidirectional search schemes (headline, 3-13x)

### The enabler: bwa's index is already bidirectional

bwa indexes S = T . revcomp(T), and revcomp(S) = S. A single BWT supports bidirectional
extension. bwt.c:262-275 `bwt_extend()` implements this:

    void bwt_extend(const bwt_t *bwt, const bwtintv_t *ik, bwtintv_t ok[4], int is_back)
    {
        bwtint_t tk[4], tl[4];
        bwt_2occ4(bwt, ik->x[!is_back] - 1, ik->x[!is_back] - 1 + ik->x[2], tk, tl);
        for (i = 0; i != 4; ++i) {
            ok[i].x[!is_back] = bwt->L2[i] + 1 + tk[i];
            ok[i].x[2] = tl[i] - tk[i];
        }
        ok[3].x[is_back] = ik->x[is_back] + (...primary correction...);
        ok[2].x[is_back] = ok[3].x[is_back] + ok[3].x[2];
        ok[1].x[is_back] = ok[2].x[is_back] + ok[2].x[2];
        ok[0].x[is_back] = ok[1].x[is_back] + ok[1].x[2];
    }

The `bwtintv_t` (bwt.h:62-64) is a 3-element interval: x[0]=forward SA start,
x[1]=backward SA start, x[2]=size. `bwt_extend` calls `bwt_2occ4` once and derives
all 4 child intervals in ~10 lines of arithmetic.

The device-side `d_bwt_2occ4` (fm_device.cuh:88) already returns all 4 counts —
it is exactly the primitive `bwt_extend` needs. Porting `bwt_extend` to device is
trivial: no second index, no extra memory, still one 2occ4 probe per node.

### What search schemes do

Backtracking costs ~3^k * C(L,k) node-pops. A search scheme partitions the read into
p parts and runs several searches, each with a staircase error budget (pi/L/U vectors)
that forces the first-searched part to have 0-1 errors — collapsing the expensive shell.

### Worked estimates for L=46, max_diff=4

**2-part pigeonhole (one page of proof, no scheme theory):**
Every <=4-error occurrence has a half with <=2 errors. Search the 3' half with e<=2
then extend backward; search the 5' half with e<=2 via forward extension then extend.
~6.5k + 6.5k = 13k nodes vs 41k -> **3.2x**.

**3-part staircase (pi=(3,2,1), U=(1,2,4) and covering partners):**
First phase = 16bp with <=1 error ~ 424 nodes; survivors ~ 72 occurrences; extension
dies within ~3 bases because interval size drops 4x per base. ~1,000 nodes per search,
a handful of searches -> **~13x**.

Kianfar/Pockrandt report 35x over backtracking for 101bp/k=2. Our k=3-4 at L=30-64
will be less, but 3-13x is the credible band — and it is multiplicative with every
memory-side optimization below, because it removes probes rather than making them cheaper.

### Device-side node format

Current stack node: `pack_node(i, mm, go, ge, st)` = 25 bits in a u32
(dfs_engine.cuh:83). Stack entry = (k:u64, l:u64, node:u32) = 20 bytes.

Bidirectional node needs the forward interval: (k_fwd, l_fwd, k_bwd, l_bwd) or the
bwtintv_t triple (x[0], x[1], x[2]). That is 3*u64 = 24 bytes per node vs 20.
Shared-memory stack at CAP_SM=512: 12 KB/warp vs 10 KB — still 8 warps/SM on sm_86
(100 KB shared). Manageable.

### Caveats

1. Published optimum schemes are Hamming distance. For edit distance, boundary slack
   is needed (cheap, since we only need a superset — false positives are reconciled).
2. max_diff varies per read (3/4/6 for L=30-91). Need schemes for k=3, 4, and
   possibly 5.
3. False-positive rate must stay near the current 0.531%, or the CPU reconcile becomes
   the bottleneck again (the CAP=640 lesson). Specificity comes from extending
   candidates to the full read inside the index before reporting has_hit=1.
4. The 2M max_entries cap means bwa itself rejects some reads the GPU would flag —
   those are already handled by the reconcile path.

### Implementation sketch

1. Port `bwt_extend` to `fm_device.cuh` as `d_bwt_extend()`.
2. Add a `bwtintv_t`-based stack node to `dfs_engine.cuh` (alongside the existing one,
   gated by a template or #ifdef).
3. Implement the 2-part pigeonhole as a new `d_dfs_has_hit_scheme()` device function:
   - Lane 0: backward search on 3' half with e<=2 (using existing DFS logic, max_diff=2).
   - Lane 1: backward search on 5' half with e<=2, then forward-extend via d_bwt_extend.
   - `__any_sync`: if either half finds a survivor, extend to full read bidirectionally.
   - Report has_hit=1 only if full-read extension succeeds (interval non-empty at i=0).
4. Validate on sub2k/sub100k against golden md5 eecf35c1.
5. Measure node-pops/read and false-positive rate.

### Literature

- Kianfar, Pockrandt, Torkamandi, Luo, Reinert — "Optimum Search Schemes for
  approximate string matching using bidirectional FM-index" (bioRxiv 2018 / thesis 2020).
  MILP-based derivation of optimal schemes; used in SeqAn library.
  https://arxiv.org/abs/1711.02035
- Kucherov, Salikhov, Tsur — "Approximate string matching using a bidirectional index"
  (CPM 2014). Combinatorial characterization of optimal search schemes.
- Renders, Depuydt, Rahmann, Fostier — "Automated design of efficient search schemes"
  (J. Comput. Biol. / RECOMB-Seq 2024). MILP-optimal schemes for up to k=7.
- Gottlieb, Reinert — "SeArcH schemes for Approximate stRing mAtching"
  (NAR Genomics and Bioinformatics, 2025). New framework extending prior results.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11915513/
- Depuydt, Fostier, Gottlieb et al. — "Search Schemes for Approximate Pattern Matching:
  An Overview" (OASIcs, 2025). Survey; discusses MILP-optimal schemes and limits.
  https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.Manzini.9
- Renders, Marchal, Fostier — "Dynamic partitioning of search patterns" (iScience 2021).
  Dynamic partitioning leveraging full bidirectional FM-index.

**No published work combines GPU + bidirectional FM-index + optimal search schemes.**
This would be a publishable result on its own.

---

## 3. Idea B — Two-level Occ layout — **DOWNGRADED to 1.16x (D7), SHELVED**

> The gain below was measured at 32 warps/SM. Best-of-3 across occupancy: 1.93x at 32
> warps/SM but only **1.16x at the shipped 8 warps/SM** — at low occupancy the memory system
> is limited by request rate/latency, not bytes. Not worth re-laying out a 3.14 GB index.

### Current layout (64 bytes per probe)

bwa's bucket (bwt.h:74, fm_device.cuh:71):

    [ 4 x uint64 absolute Occ checkpoints (32 B) ][ 8 x uint32 packed 2-bit BWT, 128 bases (32 B) ]

Every `d_bwt_2occ4` call touches one 64-byte bucket (shared-bucket fast path,
fm_device.cuh:93) or two (split path). Half of every probe is checkpoint overhead.

### Proposed two-level repack (32 bytes per probe)

Repack at load time (one-time cost, ~10 s for hs37d5):

- **Superblock** = 65,536 bases -> 4 x u64 absolute counts.
  Table size: 6.28e9 / 65536 * 32 B = **3.07 MB** — L2-resident on a 3090 (6 MB L2).
- **Block** = 32 B = 4 x u16 superblock-relative counts (8 B) + 24 B sequence (96 bases).

Result: exactly one 32-byte sector per probe instead of one 64-byte line. Index shrinks
3.14 GB -> 2.09 GB (materially better TLB coverage for a pure-scatter workload).

Refinement: store only 3 counts, derive the 4th from the block index
(cnt[3] = block_bases - cnt[0] - cnt[1] - cnt[2], where block_bases is known from the
address). Gives 104 bases/block, **1.93 GB** total.

Bit-exact by construction — identical Occ values, different encoding.

### Why the earlier "2-bit compressed BWT" idea missed this

The BWT is ALREADY 2-bit packed (bwt.h:80, `bwt_B0` macro). The win is not packing
the sequence — it is **deleting the absolute checkpoints** from the hot path and
replacing them with an L2-resident superblock table.

### Pairing with cache hints

Use `__ldcs` (streaming, non-caching) on block loads so the streaming BWT traffic does
not evict the 3 MB superblock table from L2. `__ldcs` is available on sm_80+.

### Diagnostic

FMTEST_KRANGE at ~512 MB: the fmtest.cu harness measures occ4 throughput at different
address ranges. L2-resident (3.14 GB) = 9.9 G/s; full = 2.3 G/s. A 512 MB point is
far past L2 but has 6x smaller TLB footprint — if it lands well above 2.3 G/s, the
ceiling is address translation / locality, not raw bandwidth, and Idea B's footprint
reduction matters more than its byte reduction.

---

## 4. Idea C — cnt_table -> popc — **DONE, +18%, shipped as default**

> Confirmed at +18% (bit-exact, fmtest-validated; `-DFM_OCC_CNTTABLE` reverts). Mechanism is
> constant-cache/instruction cost, *not* register pressure: popc uses 80 registers vs 79.

### The problem

`d_occ_aux4` (fm_device.cuh:58-62):

    return c_cnt_table[b & 0xff] + c_cnt_table[(b >> 8) & 0xff]
         + c_cnt_table[(b >> 16) & 0xff] + c_cnt_table[b >> 24];

`c_cnt_table` is `__constant__` (fm_device.cuh:42). Constant memory broadcasts only
when all lanes read the same address; divergent access serializes. Every lane holds a
different BWT word, so `d_occ_aux4` issues 4 divergent constant loads per u32 word,
up to 8 words per occ4 — up to **32 fully-serialized constant loads per probe**.

### The fix: bit-sliced __popc

Replace cnt_table lookups with bit-sliced population counts. bwa's 2-bit encoding
(A=0, C=1, G=2, T=3; two bits per base; 16 bases per u32):

    __device__ __forceinline__ uint32_t d_occ_aux4_popc(uint32_t b) {
        uint32_t b0 = b & 0x55555555u;        // low bits of each 2-bit symbol
        uint32_t b1 = (b >> 1) & 0x55555555u; // high bits
        uint32_t a = __popc(~b0 & ~b1);       // 00 = A
        uint32_t c = __popc( b0 & ~b1);       // 01 = C
        uint32_t g = __popc(~b0 &  b1);       // 10 = G
        uint32_t t = __popc( b0 &  b1);       // 11 = T
        return a | (c << 8) | (g << 16) | (t << 24);
    }

4 `__popc` + ~6 ALU ops, zero memory. In the 32-byte block layout (Idea B), the scan
drops to <=6 words, so <=24 popc calls per probe.

### Why the earlier "popc on bandwidth grounds" rejection was incomplete

The kernel IS bandwidth-bound — correct. But the constant-cache serialization is an
ORTHOGONAL stall: it adds instruction-issue latency on top of the memory wait. The
popc replacement eliminates the serialization stalls without changing bandwidth. The
win is from removing constant-cache miss stalls, not from reducing memory traffic.

### Diagnostic

Nsight Compute on k_dfs_warp2:

    ncu --metrics smsp__warp_issue_stalled_imc_miss_per_warp_active.pct \
        --kernel-name k_dfs_warp2 --launch-count 1 \
        ./bwa-aln-gpu -t 16 hs37d5 sub100k.fq > /dev/null

If imc_miss (constant-cache miss) is a significant stall reason, Idea C is worth doing.

---

## 5. Idea D — Lane-utilization collapse — **REJECTED (D4)**

> Premise false: measured mean active lanes 25.97/32 (81%), spills 0.000/wave. With
> CAP_SM=512 the frontier never approaches capacity, so the `room/9` throttle never engages.

### The problem

dfs_engine.cuh:127-128:

    int n_active = sp < 32 ? sp : 32;
    int r9 = room / 9; if (n_active > r9) n_active = r9; if (n_active < 1) n_active = 1;

The spill at line 118 triggers when `room < 9` and restores only CHUNK=128 entries.
For CAP_SM=512, after spill: room = 128 + (512 - sp_before_spill). The warp then
pops 1 node (n_active=1), pushes up to 9 children, room drops by 8, and it takes
~14 single-pops to re-trigger the spill. For bushy reads (the mass), the warp
oscillates in sp in [CAP-128, CAP], running at <=44% lane occupancy, degrading to
1 active lane at the trigger point.

### The fix

Change line 118 from `if (room < 9)` to `if (room < 32 * 9)` (= 288). This makes
the spill trigger proactively, keeping n_active = 32 for more waves. The cost is more
frequent spill/unspill (128-entry coalesced shared<->global copies), but those are
not HBM-scatter — they are sequential per-warp backing accesses.

Alternatively, make CHUNK adaptive: spill more entries when room is critically low.

### Diagnostic

Nsight Compute: `smsp__warps_active.avg` and the wave histogram from DFS_INSTRUMENT
(g_waves, g_pops -> mean active lanes = g_pops / g_waves).

---

## 6. Idea E — Read-level deduplication (1.2-1.5x, trivially bit-exact)

### Why it is bit-exact

bwa aln ignores base qualities entirely. The GPU path confirms this:

    aln_gpu.cu:336   c->seq_flat[...] = p->seq[j]>3?4:3-p->seq[j];   // complement, N->4
    aln_gpu.cu:335   c->rp[i].max_diff = bwa_cal_maxdiff(p->len, ...);  // length-only
    aln_gpu.cu:334   bwt_cal_width(bwt, p->len, p->seq, w.data());     // sequence-only

Identical sequences produce identical (seq_flat, w_flat, bid_flat, max_diff) ->
identical DFS trees -> identical has_hit. The CPU reconcile and output stay per-read
in original order, so drand48 ordering is untouched.

### Why it matters for aDNA

aDNA libraries duplicate heavily (PCR duplicates, adapter dimers). The
low-complexity/adapter-dimer reads that dominate duplication are also the bushiest
trees — so the saving is superlinear in the duplication rate.

### Measurement

    zcat AVA1B.combined.fq.gz | awk 'NR%4==2' | sort | uniq -c | sort -rn | head -20

### Implementation

In the preprocess loop (aln_gpu.cu:330-340): hash each read's (len, seq_flat[seq_off..
seq_off+len]), run the DFS once per distinct key, broadcast has_hit to all duplicates.
The reconcile at line 280 still runs per-read (it needs per-read bwt_match_gap for the
full alignment records), but the GPU kernel skips duplicates.

---

## 7. Idea F — Use the idle CPU as a 17th worker (~+15%, free)

### The arithmetic

CPU `bwa aln -t 16` = 4,261 reads/s. GPU = 22,445 reads/s. The CPU is busy only
~18 s of 176 s (preprocess + 0.5% reconcile). Feeding it ~15% of reads through the
same chunk queue adds ~15% end-to-end.

Wall-time model: GPU handles 85% in ~133s; CPU handles 15% in ~139s (overlapped);
wall time ~ 139s vs current 161s -> ~14% improvement.

### Why it is bit-exact

The CPU runs the reference implementation (`bwt_match_gap`), so bit-exactness is
definitional. The Phase-5 ready-queue (aln_gpu.cu:196-247) already has the right shape.

### Implementation

Add a CPU consumer thread alongside the GPU workers. It pops Chunks from the same
ready queue, runs bwt_match_gap on all reads (setting has_hit directly from n_aln > 0),
and marks the chunk done. The finisher's reconcile skips reads already processed.

---

## 8. CUDA 13.x relevance for sm_86

CUDA 13.3 (current) supports sm_86 (Ampere). Key findings from the release notes:

- **Dropped** (CUDA 13.0): Maxwell, Pascal, Volta (< sm_75). Ampere still supported.
- **No new sm_86-specific features** in CUDA 13.x. All new optimizations target
  sm_90+ (Hopper TMA, PDL) and sm_100+ (Blackwell).
- **L2 Cache Control** (sm_80+): `cudaAccessPolicyWindow` — available but limited to
  contiguous ranges (not useful for scattered root buckets; see Phase 4 negative result).
- **Async data copies** (cp.async, sm_80+): available; could pipeline rank-table loads.
- **cuSPARSE SpMVOp** (13.3): 11% avg speedup for sparse random-access patterns —
  confirms NVIDIA is optimizing scatter workloads, but not applicable to FM-index.
- **Programmatic Dependent Launch**: sm_90+ only. Not available on sm_86.
- **TMA (Tensor Memory Accelerator)**: sm_90+ only. Not applicable.

**Conclusion: no CUDA 13 feature specifically helps random-access gather patterns on
Ampere.** The optimization must come from algorithmic (Idea A) and representation
(Idea B) changes, not from new hardware features.

---

## 9. The three diagnostics — **ALREADY RUN (D1-D7)**

> Results in PROGRESS.md. Only Nsight Compute remains (`sudo pacman -S nsight-compute`).

### Diagnostic 1: Node-pop histogram by (depth, errors-used)

Confirms or kills the work-distribution model; sizes Idea A exactly.

    # Build with instrumentation
    make bwa-aln-gpu NVCCFLAGS_EXTRA="-DDFS_INSTRUMENT"
    # Run on sub100k, all reads sampled
    DFS_INSTR_MOD=1 GPUALN_HISTO=1 ./bwa-aln-gpu -t 16 hs37d5 sub100k.fq > /dev/null 2>diag1.log
    # Parse the [instr] section

### Diagnostic 2: FMTEST_KRANGE at 512 MB

Tests whether the ceiling is TLB/locality vs raw bandwidth.

    make fmtest
    # Full range (baseline):
    ./fmtest hs37d5.fa sub10k.fq 10000000
    # 512 MB range (need to add FMTEST_KRANGE env var to fmtest.cu):
    FMTEST_KRANGE=$((512*1024*1024/64)) ./fmtest hs37d5.fa sub10k.fq 10000000

If 512 MB lands well above 2.3 G/s, the ceiling is address translation, and Idea B's
footprint reduction matters more than its byte reduction.

### Diagnostic 3: Nsight Compute on k_dfs_warp2

Settles Ideas B, C, D in one run.

    ncu --metrics dram__bytes_read.sum,lts__t_sector_hit_rate_pct,\
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
    smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,\
    smsp__warps_active.avg \
    --kernel-name k_dfs_warp2 --launch-count 1 \
    ./bwa-aln-gpu -t 16 hs37d5 sub100k.fq > /dev/null

Key questions:
- dram__bytes_read.sum per node-pop: 64 B or 128 B? (Settles B)
- lts__t_sector_hit_rate: how much L2 reuse? (Settles B)
- imc_miss stall %: constant-cache serialization? (Settles C)
- smsp__warps_active.avg: mean active lanes? (Settles D)

---

## 10. Recommended execution order — **SUPERSEDED**

> Current order is in `cuda/OPTIMIZATION_IDEAS.md` (Round 3, Execution order).

### Incremental block (low-risk, bit-exact, compounding ~2.5-3x)

Run after diagnostics confirm the model:

1. **Idea C** (cnt_table -> popc): ~10-20%, 1-hour change, zero risk.
2. **Idea D** (proactive spill): 5 lines, zero risk, uncertain payoff.
3. **Idea E** (read dedup): 1.2-1.5x for aDNA, trivially bit-exact.
4. **Idea B** (two-level Occ): 1.5-2x, medium effort (repack at load time + new
   occ4 function), bit-exact by construction.
5. **Idea F** (CPU as 17th worker): ~+15%, free, pipeline already supports it.

### Parallel research track (the exponent changer)

6. **Idea A** (bidirectional search schemes): 3-13x, the only thing that changes the
   exponent. Start with the 2-part pigeonhole (simplest, smallest node expansion).
   Validate on sub2k/sub100k. Measure false-positive rate. This is the publishable result.

### CompileIQ (overnight, no code changes)

The kernel sits on a jagged compiler-control landscape: maxrregcount 48 -> 6,445 r/s,
40 -> 5,243, default 69 regs -> 9,934. NVIDIA CompileIQ
(https://github.com/NVIDIA/CompileIQ) can search this space automatically.
Requirements: clean scalar objective (reads/s on sub100k) + hard correctness gate
(md5 eecf35c1). On CUDA 13.3, use PtxasSearchSpace(version="13.3"). Expect single-digit
to ~15% — small next to Idea A, but free.

---

## 11. Summary — **STALE BASELINE (assumes 22k reads/s pre-Phase-6)**

| Idea | Mechanism | Est. speedup | Risk | Bit-exact | Effort |
|------|-----------|-------------|------|-----------|--------|
| A: Search schemes | Reduce node-pops 41k->1-13k | 3-13x | Medium (FP rate) | Yes (superset) | High |
| B: Two-level Occ | Halve bytes/probe 64->32 B | 1.5-2x | Low | Yes (repack) | Medium |
| C: popc for cnt_table | Eliminate const-cache serialization | 1.1-1.2x | None | Yes | Low |
| D: Proactive spill | Keep 32 lanes active | 1.0-1.1x | None | Yes | Trivial |
| E: Read dedup | Skip duplicate sequences | 1.2-1.5x | None | Yes | Low |
| F: CPU 17th worker | Use idle CPU cores | ~1.15x | None | Yes | Low |
| CompileIQ | Compiler flag HPO | 1.0-1.15x | None | Yes | Zero |

A is multiplicative with B+C+D+E+F. Combined realistic target: **15-40x over the
current 22k reads/s** on a single RTX 3090, i.e. **330k-880k reads/s**, vs the
16-core CPU's 4,261 r/s.
