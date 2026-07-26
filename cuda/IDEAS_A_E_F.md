# Ideas A, E and F — design document

Scope: the three optimisation ideas that survived Phase 6/7/8 measurement and are not yet
shipped. Written to be actionable: what the idea is, why it is bit-exact, what has been
measured, what remains to build, and where it can go wrong.

Status at time of writing (after Phase 6 shipped 1.82x):

| idea | estimate | measured so far | built? |
|---|---|---|---|
| **A** — bidirectional search schemes | ~2x (was 2.5–5x) | schemes implemented + priced end-to-end | probe complete, engine not built |
| **E** — read deduplication | ~10% work, 5–10% wall | full A/B on two datasets | no |
| **F** — CPU as an extra aligner worker | ~+8–13% | ratios only | no |

Authoritative measurements live in `PROGRESS.md` (D1–D7, Phase 7, Phase 8). This document
explains the *why* and the *how*; it does not restate every number.

---

## 0. The shared foundation: the GPU only owes a superset

Every idea here depends on one property of the existing architecture, so it is worth stating
precisely.

`aln_gpu.cu:251` collects every read the GPU flagged and re-runs it through bwa's own
`bwt_match_gap` on the CPU:

```c
std::vector<int> idx; for (int i=0;i<nseq;i++) if (c->has_hit[i]) idx.push_back(i);
```

The `.sai` records for those reads are therefore produced by the reference implementation, not by
the GPU. The GPU's only obligation is:

> **has_hit_gpu(r) ≥ has_hit_bwa(r), pointwise.** No false negatives.
> False positives cost CPU reconcile time, not correctness.

This is much weaker than "reproduce `bwt_match_gap`'s node set", which is how the engine was
originally framed. It is what licenses replacing the search algorithm entirely.

### Why bwa's bound is exactly edit distance

`gap_init_opt()` sets `mode |= BWA_MODE_GAPE`, and the pipeline never passes `-e`. In the engine
this reaches:

```c
int m = max_diff - (e_mm + e_go);
if (mode & BWA_MODE_GAPE) m -= e_ge;
```

so mismatches + gap opens + gap extensions all count against `max_diff`. Every other bwa
constraint (`indel_end_skip=5`, `max_del_occ`, `max_gapo`, the `bwt_cal_width` bound, and the 2 M
`max_entries` cap) only ever *removes* candidates. Therefore:

> bwa maps read r ⟹ ∃ a genome substring within **edit distance ≤ max_diff** of r.

Any search that finds all edit-distance-≤`max_diff` occurrences is a valid superset. That is the
predicate Idea A implements.

A useful corollary: the 2 M `max_entries` cap means bwa itself sometimes returns `n_aln=0` for a
read that *does* have an occurrence. A superset flags such a read, the CPU reconcile re-runs the
real `bwt_match_gap`, that also bails, and the output is identical. Bit-exactness survives.

---

## 1. Idea A — bidirectional search schemes

### 1.1 The problem being attacked

D3 measured where the work is. Per read (sub100k, `-l 1024 -n 0.01 -o 2`): **40,924 node-pops**,
~1.045 FM probes each. Distribution by errors used:

| errors | e=0 | e=1 | e=2 | e=3 | e=4 |
|---|---|---|---|---|---|
| share of pops | 0.03% | 1.6% | **37.5%** | **59.4%** | 1.5% |

with the depth mass at 11–18 bases consumed, peaking at 14. An analytic model
`N(j) ≈ [Σ_{e≤E(j)} C(j,e)·3^e] × min(1, 6.28e9/4^j)` predicts 41,000 against 40,924 measured.

The reading: at depths 11–18 essentially every k-mer exists in a 3.1 Gbp genome, so nothing is
pruned by the index. The tree is a **combinatorial shell** — it is `3^e·C(j,e)` growing faster than
the genome can kill it. `e≥2` is 97% of all work.

This is why every representation-side idea failed. k-mer existence bitmaps, ftab/jump tables,
L2-resident prefix tables and singleton-interval collapse all target depths that hold almost no
work. The only lever that touches a combinatorial shell is **reducing the exponent**, i.e. never
letting the search carry 2–3 errors through the explosive depth band in the first place.

### 1.2 What a search scheme is

Partition the read into `p` contiguous **parts**. A **search** is a triple `(π, L, U)`:

- `π` — the order in which parts are visited. Must be **contiguous**: the set of visited parts is
  always an interval, so the search grows outward from an anchor, one base at a time, left or right.
- `L`, `U` — lower and upper bounds on the *cumulative* error count after each part is completed.

A **search scheme** is a set of searches. It is **covering** for `k` errors if, for every
distribution `(e_1..e_p)` of at most `k` errors across the parts, some search admits it.

Two consequences matter here:

1. **The union of a covering scheme is exact, not approximate.** Every occurrence with ≤k errors is
   found by at least one search, and no search ever reports an occurrence with >k errors. So the
   union is precisely the ≤k-error occurrence set — no false negatives, and against bwa it is a
   strict superset (§0), which is the safe direction.
2. **The staircase in `U` is where the saving comes from.** A search whose first part must carry 0
   or 1 errors never enters the `e=2,3` shell during the depth band where that shell is enormous.
   The expensive part of the tree is deferred until the interval has already collapsed.

### 1.3 The enabler: bwa's index is already bidirectional

Search schemes need extension in both directions. Normally that means a second index over the
reversed text. It is not needed here.

bwa builds its BWT over `S = T · revcomp(T)` (hence `fm.seq_len` = 6.28e9 = 2 × 3.14e9). That text
satisfies `revcomp(S) = S`, so a single index supports bidirectional extension. bwa-mem already
exploits this — `bwt.c:262 bwt_extend()` is exactly this operation, used by the SMEM code.

**So Idea A costs no extra index and no extra GPU memory.** This was the single most important
finding in scoping it.

State is a triple (mirroring `bwtintv_t.x[3]`):

| field | meaning |
|---|---|
| `x0` | SA interval start of the matched pattern `P` |
| `x1` | SA interval start of `revcomp(P)` |
| `x2` | interval size (shared) |

The device mirror is built and validated (`fm_device.cuh`: `bwtintv_dev`, `d_bwt_extend`,
`d_bwt_set_intv`). Semantics, taken from `bwt_smem1a`:

- `is_back=1` **prepends** a base to `P`; drives on `x0`, and recovers `x1` by cumulative sums.
- `is_back=0` **appends**; drives on `x1`. Appending base `c` is the same as prepending
  `complement(c)` to `revcomp(P)`, so the caller must select **`ok[3-c]`**, not `ok[c]`. This is
  easy to get wrong and is the reason Test C exists.

`fmtest` Test C replays a random alternating forward/backward walk on device and on host with
bwa's own `bwt_extend`, comparing the full triple at every step:
**PASS — 20,000 queries, 300,630 steps, 0 mismatches.**

### 1.4 Measured costs

Two independent probes, because neither alone is sufficient.

**Phase 7 — `DFS_STAIR` (authoritative for absolute cost).** Adds a staircase budget to the *real*
production engine, pruning at child emission so `g_pops` reports the true tree size. Because it
runs inside the real engine it includes gaps and the `bwt_cal_width` bound. Cost of **one** search
as a fraction of the current full backtrack:

| p:U1:U2 | sub100k | 2730 short |
|---|---|---|
| 2:2:4 | 44.59% | 26.89% |
| **3:1:2** | **7.09%** | **2.76%** |
| 4:1:2 | 19.87% | 8.66% |
| 5:0:1 | 3.34% | 1.01% |

Cross-check: the 2-part figure (44.59%) matches the 39% `e≤2` fraction predicted independently
from the D3 error histogram. Model and instrumentation agree.

**Phase 8 — `bidir_cost` (authoritative for the anchor ratio).** A standalone warp-cooperative
bidirectional staircase search with a configurable anchor. The question it answers: are searches
anchored at an *interior* part more expensive than end-anchored ones? That was the last unmeasured
term in the estimate.

| anchor | 5′ end | 1/4 | middle | 3/4 | 3′ end |
|---|---|---|---|---|---|
| pops/read | 853 | 851 | 854 | 856 | 856 |

**Anchor position is irrelevant (≤0.4% spread), and this holds across every configuration tried
(1–3% spread).** Pure-forward, bidirectional and pure-backward searches cost the same. Tree size is
governed by the staircase budget and genome statistics, not by direction.

> **Do not quote `bidir_cost` ratios as speedups.** It is mismatch-only and has no `bwt_cal_width`
> bound, so its "full" baseline (105,030 pops/read) is 2.6x *larger* than production's 40,924.
> Ratios against it are inflated. Its valid contribution is the anchor comparison, which is
> internal to one probe. Absolute per-search costs come from `DFS_STAIR`.

**Resulting estimate — SUPERSEDED by direct measurement (Phase 11).** The published MIP-optimal
schemes (Kianfar et al. Table 3) need only **3 searches**, not 5–8, and have been implemented with
full (π, L, U) semantics and priced. Measured against backtracking with the two-sided bound on
both sides — the only comparison that matters, since the shipped engine already has bwa's bound:

| K | P=K+1 | P=K+2 |
|---|---|---|
| 3 | 1.40x | **2.97x** |
| 4 | 2.55x | **2.70x** |

Against a bound-free baseline the same schemes look like 9–48x, but that double-counts: the width
bound and the scheme prune much the same nodes. Discount further for edit vs Hamming (~0.65x per
the paper's own Table 2) and **~2x is the realistic expectation**.

For the record, both earlier estimates were too high. The original "3–13x" priced a *single*
search and forgot a covering scheme needs several; the revised "2.5–5x" was right only at its
lower bound, and only before the Hamming/edit discount.

### 1.5 The hard part: designing the covering scheme

A monotone staircase is a *cost model*, not a covering scheme. Worked counter-example, p=3, k=4,
U=(1,2,4):

- distribution `(0,4,0)`;
- anchor at part 1 → cumulative after part 2 is 4, cap is 2 ✗;
- anchor at part 3 → symmetric ✗;
- anchor at part 2 → needs `e_2 ≤ U_1 = 1`, but `e_2 = 4` ✗.

Uncovered ⇒ a read whose only alignment has all four errors in the middle third would be missed.
That is a **false negative**, the one failure mode that breaks bit-exactness.

Pigeonhole forces `p ≥ k+1` (with k+1 parts some part is error-free). But even at p=5 a plain
staircase is not automatically covering, because `π` must stay **contiguous**: the heavy part
cannot always be deferred to last. `(0,4,0,0,0)` anchored at part 3 must reach part 2 before part 1,
and the cumulative cap at that position is below 4.

This is exactly why the literature solves scheme design with a mixed-integer program rather than by
hand (Kianfar/Pockrandt/Reinert; Renders/Fostier; Gottlieb/Reinert). **The next step is to adopt a
published covering scheme for k=3/4/6 and price it with the per-search costs above — not to invent
one.** Hand-rolled schemes are the most likely source of a silent false negative in this project.

Note `max_diff` varies per read: 3 at L=30, 4 at L=46, 6 at L=91. The pipeline's short/long split
at 64 bp means the GPU sees roughly k=3–4, but a scheme is needed for each k in range.

### 1.6 Implementation plan

**Node format.** Interval triple + packed metadata:

| field | bits/bytes | note |
|---|---|---|
| `x0`, `x1` | 8 B each | interval starts |
| `x2` | 4 B | size — max single-base interval ≈1.57e9 < 2³², so u32 is safe |
| packed | 4 B | `lo`(8) `hi`(8) `n_mm`(5) `n_gapo`(3) `n_gape`(4) `state`(2) = 30 bits |

**24 B/node** vs the current 20 B. At `CAP_SM=512` that is 12 KB/warp, 48 KB/block at wpb=4,
96 KB for 2 blocks/SM — inside the 100 KB sm_86 limit. **Occupancy stays at 8 warps/SM**, which
D6/D7 showed is the sweet spot. (An earlier estimate of 28 B forcing CAP_SM=384 was pessimistic;
storing `x2` as u32 avoids it.)

**Engine shape.** Reuse the Phase-6 structure verbatim — warp-cooperative, one read per warp,
two-pass child generation (count → warp prefix-sum → emit direct to shared), zero local memory.
Those fixes were worth 1.82x and are orthogonal to the search algorithm.

Per read, loop over the scheme's searches and early-exit the whole read on the first hit, since
`has_hit` is a disjunction over searches. The ~99.5% of reads that never hit pay for all searches;
the ~0.5% that do hit usually exit on the first.

Direction is derived from the node, not stored: extend right while `hi < right_target`, then left
while `lo > 0` (the `bidir_cost` kernel already does this). The per-search `U` vector is indexed by
how many parts are complete, computed from `lo`/`hi` against the part boundaries.

**Gap handling.** Keep bwa's `max_gapo`/`max_gape` limits — any alignment bwa finds obeys them, so
respecting them loses nothing. **Drop `indel_end_skip` and `max_del_occ`**: both only remove
candidates, so dropping them is more permissive and stays a superset. `max_del_occ` in particular
depends on the interval size at a given traversal point, which differs between search orders and
therefore cannot be replicated faithfully anyway.

**Integration.** The scheme engine replaces the `has_hit` computation only. Everything downstream —
the flagged-read list, the multithreaded `bwt_match_gap` reconcile, the ordered finisher, samse — is
untouched, so bit-exactness is structural rather than something to re-verify from scratch. Validate
as usual: sub2k/sub10k/sub100k md5s, then the 3.5 M and 671 k sets against their CPU goldens.

### 1.7 Open problems and risks

1. **Loss of the `bwt_cal_width` bound — the biggest unknown.** bwa precomputes, for each prefix of
   the read, a lower bound on the errors needed for that prefix to occur, and prunes on it. It is
   directional: it applies to the backward search only. Interior-anchored searches cannot use it.
   The `bidir_cost` probe suggests this bound is worth about **2.6x** (105,030 vs 40,924 pops/read,
   though that comparison also involves gaps). If a scheme's searches lose it, some of the scheme's
   advantage is eaten. Mitigation: precompute a per-direction bound array on the CPU
   (`bwt_cal_width` on the reversed read gives the forward-direction bound); cost is a second
   `bwt_cal_width` pass in preprocessing, currently only ~1.4 s of a 176 s run. **This should be
   measured before committing to the full engine.**
2. **False-positive rate.** The scheme's superset must not flag many more reads than today's 0.52%,
   or the CPU reconcile becomes the bottleneck — this already happened once (Phase 2, CAP=640,
   4.6% flagged → 84 s CPU tail dwarfing a 3.4 s kernel). Dropping `indel_end_skip`/`max_del_occ`
   widens the superset slightly. Measure the flag rate early.
3. **Scheme correctness.** A non-covering scheme produces silent false negatives — no crash, just a
   subtly different `.sai`. Mitigation: verify the covering property exhaustively offline (enumerate
   all `(e_1..e_p)` with sum ≤ k and assert each is admitted by some search) as a unit test, and
   keep the full-file byte-identical validation as the backstop.
4. **Per-read `max_diff` variation** means either several schemes or one scheme for the worst case.
5. **Register pressure.** The bidirectional node carries an extra u64 through the hot loop. Phase 6
   got the kernel to 0 B local memory at 79–80 registers; regressing to local memory would cost more
   than the scheme gains. Watch `ptxas -v` on every iteration — this is the single cheapest guard
   rail in the project.

---

## 2. Idea E — read-level deduplication

### 2.1 The idea and why it is bit-exact

`bwa aln` never looks at base qualities (only `-q` trimming does, before alignment). Two reads with
identical sequences therefore produce identical `max_diff`, identical `bwt_cal_width` arrays, and
identical DFS trees. So the DFS can be run once per **distinct sequence** and `has_hit` broadcast to
every read carrying it.

Bit-exactness is trivial here because dedup touches only a boolean. The CPU reconcile still runs
per original read, in original order, so the `drand48` sequence in `bwa_aln2seq_core` and the output
order are untouched.

### 2.2 What was measured

| set | reads | distinct | dup rate | node-pops full → dedup | kernel s full → dedup |
|---|---|---|---|---|---|
| 2730 short | 671,652 | 603,472 | 10.15% | 46.60e9 → 41.87e9 (−10.17%) | 24.3 → 23.1 (−4.9%) |
| ENNBN7 | 3,505,813 | 3,136,527 | 10.53% | 94.66e9 → 84.86e9 (−10.35%) | 63.5 → 57.4 (−9.6%) |

**A hypothesis that turned out to be false.** The original estimate of 1.2–1.5x assumed duplicates
would be dominated by low-complexity reads, which were assumed to have huge DFS trees — so the work
saved would be superlinear in the read count saved. ENNBN7's top duplicates *are* low-complexity
(`CACACACA…` ×84, `TGTGTG…` ×79), yet work saved (10.17% / 10.35%) tracks read-count reduction
(10.15% / 10.53%) almost exactly. Those reads do not have oversized trees — plausibly because a
low-complexity read matches the reference immediately and exits early, rather than exhausting a
bounded tree. **Idea E is a ~10% lever, not a 1.5x one.**

Wall-clock saving is lower than work saving (−4.9% on 2730) because removing reads also removes
parallel work near the tail, where the GPU is not fully occupied.

### 2.3 Implementation sketch

Dedup within a chunk (`0x40000` reads), not globally — keeps memory bounded and preserves the
streaming structure.

1. After preprocessing a chunk, hash each read's 2-bit-packed sequence (a 64-bit hash plus a length
   field; verify on collision by comparing bytes).
2. Build `unique[]` and a `rep[]` map from read index → representative index.
3. Upload only `unique[]` to the GPU; the `ReadParam` array shrinks accordingly.
4. After the kernel, expand: `has_hit[i] = has_hit_unique[rep[i]]`.
5. Reconcile and output unchanged, over all original reads in order.

Cost: one hash pass over ~262 k reads per chunk (trivially parallel over `nT` threads) and one
`std::vector<int>` of chunk size. Note the preprocessing (`bwt_cal_width`) currently runs *before*
dedup; running it only on unique reads saves a little more CPU too.

### 2.4 Verdict

Real, cheap, low-risk, and bit-exact by construction — but it is a 5–10% wall-clock lever. Worth
doing when convenient; not worth doing before Idea A.

---

## 3. Idea F — use the idle CPU as an extra aligner

### 3.1 The idea

The pipeline (Phase 5) has three stages: a reader/preprocessor, N GPU workers, and one ordered
finisher doing the CPU reconcile plus output. The 16 CPU cores are busy only for preprocessing
(~1.4 s of a 176 s run) and the reconcile of ~0.5–2.6% of reads. They are otherwise idle while the
GPU does all the alignment.

Since `bwt_match_gap` on the CPU *is* the reference implementation, any read it aligns is
bit-exact by definition. So a CPU consumer can pull a share of reads from the same queue.

### 3.2 Revised expectation

Measured throughputs on the same box:

| set | CPU (16 threads) | GPU kernel | ratio |
|---|---|---|---|
| 2730 short (671,652 reads, 30–63 bp) | 235 s = 2,858 r/s | 23.7 s = 28,340 r/s | 9.9 : 1 |
| ENNBN7 (3,505,813 reads, 30–34 bp) | 470 s = 7,459 r/s | 62.4 s = 56,206 r/s | 7.5 : 1 |

If the CPU aligns its own share concurrently, the combined rate is `1 + 1/ratio`, i.e.
**+10% (2730) to +13% (ENNBN7)** before contention. Subtract the CPU time already spent on
preprocessing and reconcile — heavier on ENNBN7, where 23.7% of reads are reconciled — giving a
realistic **+8–13%**, dataset-dependent.

Note this estimate *fell* after Phase 6: the original "+15%" assumed a 5:1 GPU:CPU ratio, but making
the GPU 1.82x faster made the CPU proportionally less useful. Idea F is worth less now precisely
because the kernel work succeeded — and it will shrink again if Idea A lands.

### 3.3 Implementation sketch

The existing bounded "ready" queue already has the right shape:

1. Add a CPU worker stage that pops chunks from the same queue as the GPU workers, aligns every read
   with `bwt_match_gap` across `nT_cpu` threads, marks `has_hit` (and stores the resulting
   `bwt_aln1_t` so the finisher need not redo it), then hands the chunk to the ordered finisher.
2. Give the CPU stage a **smaller granularity** than a full `0x40000` chunk — at a ~10:1 ratio a CPU
   chunk should be ~1/10 the size, or the tail imbalance eats the gain. Sub-chunking is the main
   piece of new machinery.
3. Keep exactly one ordered finisher so `drand48` order and output order are unchanged.
4. Reserve threads: the finisher's reconcile and the reader's preprocessing need cores too.
   Oversubscribing will slow the GPU path by starving its host thread.

Bit-exactness: the CPU path is the reference; the GPU path is already verified; the finisher remains
a single ordered consumer. Nothing about the `.sai` byte stream changes.

### 3.4 Risks

- **Tail imbalance** is the main one; a whole chunk handed to the CPU takes ~10x longer than on the
  GPU and stalls the ordered finisher behind it. Sub-chunking is not optional.
- **Core contention** with preprocessing and reconcile can make the net gain negative on
  reconcile-heavy datasets (ENNBN7 at 23.7% is the stress case).
- Gains shrink as the GPU gets faster. If Idea A lands at 3x, the ratio becomes ~30:1 and Idea F is
  worth ~3%. **Sequence it accordingly: F is worth more now than it will ever be again, but it is
  still the smallest of the three.**

---

## 4. Recommended sequencing

1. **A, step 1 — measure the width-bound loss.** The single largest uncertainty (§1.7.1). Cheap:
   extend `bidir_cost` with a directional bound and re-measure. Do this before building the engine.
2. **A, step 2 — adopt a published covering scheme** for k=3/4/6 and unit-test the covering property
   exhaustively offline.
3. **A, step 3 — build the per-search engine** on the Phase-6 skeleton; validate md5s at every step.
4. **E** — ~10%, low risk, can be done independently at any time.
5. **F** — ~+8–13% today, shrinking; needs sub-chunking to be safe.

Independent of all three: `sudo pacman -S nsight-compute` would settle the ~2.4x gap between the
kernel (1.7 G-occ4/s) and the Occ primitive at matched occupancy (4.14 G/s), which is not bandwidth,
not occupancy, not the Occ code and not local memory. That gap is worth more than E and F combined.
