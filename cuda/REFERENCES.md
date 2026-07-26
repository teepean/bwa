# CUDA references consulted (keep checking these as tuning proceeds)

Per user request, re-check CUDA docs and NVIDIA developer forums periodically during optimization.

## Primary docs
- CUDA C++ Programming Guide (13.x): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Ampere GPU Architecture Tuning Guide (sm_86): https://docs.nvidia.com/cuda/ampere-tuning-guide/
  - sm_86: **48 max resident warps/SM**, 64K 32-bit regs/SM, ≤255 regs/thread, 100 KB shared/SM.
  - full 48 warps needs ≤ ~42 regs/thread (we use 69 -> 7 blocks = 28 warps).
- Best Practices Guide / CUDA warp-level primitives: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
- L2 persistence, Cooperative Groups, Dynamic Parallelism: in the Programming Guide special-topics.

## Directly applicable findings
- **Volkov, "Understanding Latency Hiding on GPUs"** (UCB EECS-2016-143): latency is hidden by
  ILP *and* occupancy; a dependent-chain kernel (ILP≈1) leans on occupancy. But our measurements
  show raising occupancy *hurts* -> we are stack-memory-capacity bound, not latency bound.
- **"High-Performance N-Queens Solver on GPU: Iterative DFS with Zero Bank Conflicts"**, arXiv
  2511.12009 (2025): iterative DFS with the **stack mapped to shared memory**, bank-conflict-free.
  Target structure for our next big lever.
- NVIDIA/GTC fundamental-optimization decks + forum consensus: **one-thread-per-task tree search
  scales poorly (load imbalance + warp divergence)**; prefer **assigning a subtree to a group of
  threads sharing fast memory**; keep DFS stack in shared/registers (global ~290 cyc vs L1 ~33 cyc).
- Persistent threads >> dynamic parallelism for irregular work (GTC persistent-threads study);
  device-side cudaDeviceSynchronize removed in CUDA 12.

## Phase 6 literature — search schemes (the algorithmic lever)

- **Kianfar, Pockrandt, Torkamandi, Luo, Reinert** — "Optimum Search Schemes for
  approximate string matching using bidirectional FM-index" (bioRxiv 2018 / FU Berlin
  thesis 2020). MILP-based derivation of optimal schemes; implemented in SeqAn.
  https://arxiv.org/abs/1711.02035
- **Kucherov, Salikhov, Tsur** — "Approximate string matching using a bidirectional
  index" (CPM 2014). Combinatorial characterization of optimal search schemes.
- **Renders, Depuydt, Rahmann, Fostier** — "Automated design of efficient search
  schemes for lossless approximate pattern matching" (J. Comput. Biol. 2024 /
  RECOMB-Seq 2024). MILP-optimal schemes for up to k=7; co-optimal scheme enumeration.
- **Gottlieb, Reinert** — "SeArcH schemes for Approximate stRing mAtching"
  (NAR Genomics and Bioinformatics, 2025). New search-scheme framework.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11915513/
- **Depuydt, Fostier, Gottlieb et al.** — "Search Schemes for Approximate Pattern
  Matching: An Overview" (OASIcs Manzini Festschrift, 2025). Survey of the field;
  discusses MILP limits and practical schemes.
  https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.Manzini.9
- **Renders, Marchal, Fostier** — "Dynamic partitioning of search patterns for
  approximate pattern matching using search schemes" (iScience, 2021). Dynamic
  partitioning leveraging full bidirectional FM-index.
- **Pockrandt** — "Approximate string matching: improving data structures and
  algorithms" (FU Berlin PhD thesis, 2019). Covers bidirectional FM-index
  implementation (constant-time) and approximate matching.

## Phase 6 literature — GPU FM-index + random memory access (2024-2026)

- **Chacón, Marco-Sola, Espinosa et al.** — "Boosting the FM-index on the GPU:
  Effective techniques to mitigate random memory access" (IEEE/ACM TCBB, 2014).
  Foundational: tiling, thread coarsening, warp-synchronous rank. PMID 26451818.
- **Han, Kim, Park, Lee** — "G³SA: A GPU-Accelerated Gold Standard Genomics Library
  for End-to-End Sequence Alignment" (ACM ICS 2025). Optimizes FM-index SMEM
  construction; minimizes random global memory accesses.
- **Zhang, Li, Meng, Zhang, Tan** — "Faster and Cheaper: Pushing the Sequence
  Alignment Throughput with Commercial CPUs" (ACM ICS 2026). OCC sampling strategies;
  references GPU random-access bottleneck.
- **Langarita, Armejach, Setoain et al.** — "Compressed Sparse FM-index: Fast
  sequence alignment using large k-steps" (IEEE/ACM TCBB, 2020). 2-step sampled
  FM-index reduces rank queries.
- **Groot Koerkamp** — "QuadRank: Engineering a High Throughput Rank" (arXiv
  2602.04103, 2026 / WABI 2026). Optimal rank data structure for FM-index;
  references GPU boosting techniques.
- **Kallenborn** — "High-performance processing of Next-Generation Sequencing data
  on CUDA-enabled GPUs" (PhD thesis, U Mainz, 2024). Comprehensive survey of GPU
  NGS including FM-index memory patterns.
- **Mognol** — "Acceleration of bioinformatics algorithms on a Processing-in-Memory
  architecture" (thesis, 2025). Demonstrates FM-index incompatibility with PIM due
  to random access; motivates GPU caching approaches.
- **Kaplan, Schmelzle, Gu et al.** — "PangenomicsBench: A Benchmark Suite and
  Characterization of Pangenomics" (IEEE ISPASS 2025). BWT-based FM-index is memory
  bandwidth-intensive due to pseudo-random access; includes GPU experiment scripts.

## Phase 6 — CUDA 13.x for sm_86

- **CUDA Toolkit 13.3 Release Notes**: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
  - Dropped Maxwell/Pascal/Volta (< sm_75) in 13.0. Ampere (sm_80/86) still supported.
  - No new sm_86-specific features; all new optimizations target sm_90+ / sm_100+.
  - L2 Cache Control (cudaAccessPolicyWindow): available on sm_80+, contiguous ranges only.
  - Async data copies (cp.async): available on sm_80+.
  - Programmatic Dependent Launch: sm_90+ only. TMA: sm_90+ only.
  - cuSPARSE SpMVOp (13.3): 11% avg speedup for sparse random-access patterns.
- **NVIDIA CompileIQ**: https://github.com/NVIDIA/CompileIQ
  - HPO for ptxas/nvcc compiler flags. Clean scalar objective + correctness gate.
  - Relevant: maxrregcount 48->6445, 40->5243, default 69->9934 r/s (jagged landscape).

## Open questions to take to the forums next
- Best shared-memory stack window size vs occupancy trade for sm_86 with ~46 bp reads.
- Warp-cooperative FM-index backtracking: split the 4-symbol Occ probe across lanes vs
  one-read-per-lane-but-grouped; any published BWA-on-GPU warp schemes beyond BarraCUDA/NVBIO.
- **NEW**: Has anyone implemented bidirectional FM-index extension on GPU? The bwt_extend
  primitive is trivial to port; the question is whether the bwtintv_t triple (24 B/node)
  fits shared-memory budgets at useful occupancy on sm_86.
- **NEW**: Two-level Occ encoding (superblock u64 + block u16-relative): any prior art
  beyond the compressed FM-index literature? The 3 MB superblock table is L2-resident.
