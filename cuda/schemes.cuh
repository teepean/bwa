/* Search-scheme tables for the bidirectional GPU engine -- part of the bwa `gpualn` port.
   Copyright (C) 2026  teepean  <https://github.com/teepean>

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>. */

/* A search scheme partitions the read into P equal parts and runs several searches, each with a
 * visiting order pi and cumulative error bounds L (lower) and U (upper) checked at part
 * boundaries. A scheme is COVERING for K errors if every distribution of <=K errors across the
 * parts is admitted by some search; the union of a covering scheme therefore finds exactly the
 * <=K-error occurrence set -- no false negatives.
 *
 * Which family matters: Kianfar's k>=3 tables are "best found in 2 hours", not proven optimal,
 * and measure 2.1-2.7x here; the Renders/SeqAn3 tables measure 14.7-16.6x at K=4 on the same
 * probe and baseline (PROGRESS.md Phase 12), because they keep the FIRST-searched part at
 * L=U=0/1 and defer the full budget to the last, already-narrowed part -- and this kernel's cost
 * is the e>=2 shell (97% of node-pops, D3).
 *
 * Sources: K=1,2 Kianfar et al. (arXiv:1711.02035, Table 3); K=3 SeqAn3
 * optimum_search_scheme<0,3>; K=4 Renders et al. (Columba search_schemes/multiple_opt/4,
 * scheme1). All stored 0-based with contiguous pi.
 *
 * K>=5 is deliberately absent: no validated table is available (SeqAn3 stops at 3, Columba ships
 * only even k for the optimum family), so those reads fall back to the exact backtracking engine.
 * For the production short branch (L<=63) only K=3 and K=4 occur, so the fallback is never hit
 * there. See sch_have().
 */
#ifndef SCHEMES_CUH
#define SCHEMES_CUH

#include <stdint.h>

#define SCH_MAXK 4
#define SCH_MAXP 8
#define SCH_MAXS 8

struct SchemeTab {
	int P, ns;                                   /* parts, number of searches */
	uint8_t pi[SCH_MAXS][SCH_MAXP];
	uint8_t L [SCH_MAXS][SCH_MAXP];
	uint8_t U [SCH_MAXS][SCH_MAXP];
};

#ifdef FM_DEVICE_DEFINE_CONST
__constant__ SchemeTab c_sch[SCH_MAXK + 1];
#else
extern __constant__ SchemeTab c_sch[SCH_MAXK + 1];
#endif

/* ---- host-side table definition + validation ----
 * NOT guarded by __CUDA_ARCH__: nvcc parses host function bodies during the DEVICE pass too, so
 * hiding these there makes every host call site fail to compile. */
#include <cstdio>
#include <cstring>
#include <vector>

struct SchRaw { int K, P, ns; const char *pi[SCH_MAXS]; const char *L[SCH_MAXS]; const char *U[SCH_MAXS]; };

/* strings are 0-based part indices, one char per part */
static const SchRaw SCH_RAW[] = {
	/* K=1, p=2, 2 searches -- Kianfar */
	{1,2,2, {"01","10"},
	        {"00","00"},
	        {"01","01"}},
	/* K=2, p=3, 3 searches -- Kianfar */
	{2,3,3, {"012","210","120"},
	        {"002","000","011"},
	        {"012","022","012"}},
	/* K=3, p=5, 4 searches -- SeqAn3 optimum_search_scheme<0,3> */
	{3,5,4, {"43210","23410","12340","01234"},
	        {"00000","00111","00022","00003"},
	        {"00333","01123","01223","02233"}},
	/* K=4, p=5, 5 searches -- Renders et al. multiple_opt/4 scheme1 */
	{4,5,5, {"01234","12034","21034","34210","43210"},
	        {"00222","00000","01111","00003","01114"},
	        {"02244","01244","01244","01444","01444"}},
};

/* Exhaustive covering + contiguity check. Returns 0 on success. Run at startup: a non-covering
 * table yields silent FALSE NEGATIVES (a subtly wrong .sai), not a crash. */
static int sch_validate(const SchRaw *r)
{
	int K = r->K, P = r->P, bad = 0;
	for (int s = 0; s < r->ns; ++s) {                       /* pi must be contiguous */
		int seen[SCH_MAXP] = {0};
		for (int j = 0; j < P; ++j) {
			int p = r->pi[s][j] - '0';
			if (p < 0 || p >= P) { fprintf(stderr, "[scheme] K=%d search %d: bad part %d\n", K, s, p); return 1; }
			seen[p] = 1;
			int lo = -1, hi = -1, cnt = 0;
			for (int t = 0; t < P; ++t) if (seen[t]) { if (lo < 0) lo = t; hi = t; ++cnt; }
			if (hi - lo + 1 != cnt) { fprintf(stderr, "[scheme] K=%d search %d: pi not contiguous at step %d\n", K, s, j); return 1; }
		}
	}
	/* every error distribution summing to <=K must be admitted by some search */
	std::vector<int> d(P, 0);
	long total = 0, uncovered = 0;
	for (;;) {
		int sum = 0; for (int i = 0; i < P; ++i) sum += d[i];
		if (sum <= K) {
			++total;
			int ok = 0;
			for (int s = 0; s < r->ns && !ok; ++s) {
				int c = 0, good = 1;
				for (int j = 0; j < P && good; ++j) {
					c += d[r->pi[s][j] - '0'];
					if (c < r->L[s][j] - '0' || c > r->U[s][j] - '0') good = 0;
				}
				ok = good;
			}
			if (!ok) { ++uncovered; if (uncovered <= 3) {
				fprintf(stderr, "[scheme] K=%d UNCOVERED distribution:", K);
				for (int i = 0; i < P; ++i) fprintf(stderr, " %d", d[i]);
				fprintf(stderr, "\n"); } }
		}
		int i = 0;                                          /* odometer over (K+1)^P */
		for (; i < P; ++i) { if (++d[i] <= K) break; d[i] = 0; }
		if (i == P) break;
	}
	if (uncovered) { fprintf(stderr, "[scheme] K=%d NOT COVERING (%ld/%ld)\n", K, uncovered, total); return 1; }
	bad = 0;
	return bad;
}

/* build the device tables; returns 0 on success */
static int sch_upload(int verbose)
{
	SchemeTab h[SCH_MAXK + 1];
	memset(h, 0, sizeof h);
	for (unsigned i = 0; i < sizeof(SCH_RAW)/sizeof(SCH_RAW[0]); ++i) {
		const SchRaw *r = &SCH_RAW[i];
		if (sch_validate(r)) return 1;
		SchemeTab &t = h[r->K];
		t.P = r->P; t.ns = r->ns;
		for (int s = 0; s < r->ns; ++s)
			for (int j = 0; j < r->P; ++j) {
				t.pi[s][j] = (uint8_t)(r->pi[s][j] - '0');
				t.L [s][j] = (uint8_t)(r->L [s][j] - '0');
				t.U [s][j] = (uint8_t)(r->U [s][j] - '0');
			}
		if (verbose) fprintf(stderr, "[scheme] K=%d: p=%d, %d searches, covering verified\n", r->K, r->P, r->ns);
	}
	if (cudaMemcpyToSymbol(c_sch, h, sizeof h) != cudaSuccess) return 1;
	return 0;
}

static inline int sch_have(int K) { return K >= 1 && K <= SCH_MAXK; }

#endif /* SCHEMES_CUH */
