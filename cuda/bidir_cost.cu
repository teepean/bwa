/* Bidirectional search-scheme COST PROBE -- part of the bwa `gpualn` GPU port (Phase 8 / Idea A).
   Copyright (C) 2026  teepean  <https://github.com/teepean>
   Derived from bwa (Heng Li; Broad Institute / Dana-Farber / Genome Research Ltd.).

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

/* WHY THIS EXISTS
 * The DFS_STAIR probe (PROGRESS.md Phase 7) measured the cost of ONE staircase search, but only
 * for searches anchored at the 3' END of the read -- the direction the production engine already
 * runs. A real search scheme also needs searches anchored at INTERIOR parts, which require true
 * bidirectional extension. Those were the one unmeasured term in the 2.5-5x estimate.
 *
 * This tool measures node-pop counts for a single staircase search with a configurable ANCHOR,
 * so end-anchored and middle-anchored searches can be compared apples-to-apples.
 *
 * It is a COST PROBE, not an aligner: mismatches only (no gaps), and it reports tree sizes, not
 * alignments. Gaps add a roughly constant factor to every search, so the RATIO between anchor
 * positions -- the thing being measured -- is what matters.
 *
 * Build: make bidir_cost
 * Run:   ./bidir_cost <ref.fa> <reads.fq> [max_err] [parts] [u1] [u2] */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

extern "C" {
#include "bwt.h"
}

#define FM_DEVICE_DEFINE_CONST
#include "fm_device.cuh"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
	exit(1); } } while (0)

#define CAPSM 384
#define MAXRD 128   /* max read length for the bound arrays */
#define WPB   4

/* node: bidirectional interval triple + packed (lo, hi, nerr) */
struct BNode { uint64_t x0, x1, x2; uint32_t p; };
__device__ __forceinline__ uint32_t bpack(int lo, int hi, int e)
{ return (uint32_t)lo | ((uint32_t)hi << 8) | ((uint32_t)e << 16); }

/* staircase cap as a function of how much of the read is matched -- identical in form to the
 * DFS_STAIR probe, so end-anchored numbers here are directly comparable to Phase 7. */
__device__ __forceinline__ int stair_cap(int matched, int len, int parts, int u1, int u2, int maxe)
{
	int b1 = len / parts, b2 = (2 * len) / parts;
	if (matched <= b1) return u1;
	if (matched <= b2) return u2;
	return maxe;
}

/* Two-sided admissible bound, the bidirectional generalisation of bwa's bwt_cal_width.
 * bidL[x] = min errors for seq[0..x-1] to occur;  bidR[x] = min errors for seq[x..len-1].
 * Greedy maximal chopping: if a string occurs with e errors, deleting the e error positions
 * leaves e+1 error-free pieces that each occur in the genome, and greedy chopping is minimal,
 * so (pieces-1) <= e. Errors in the two disjoint unmatched flanks add, hence at a node with
 * window [lo,hi) we may prune when  errors_so_far + bidL[lo] + bidR[hi] > max_err.
 * Lane 0 builds bidL by APPENDING (forward extension), lane 1 builds bidR by PREPENDING. */
__device__ __forceinline__ void d_build_bounds(fmidx_dev fm, const uint8_t *q, int len,
                                               uint8_t *bidL, uint8_t *bidR, int lane)
{
	if (lane == 0) {
		bidL[0] = 0;
		int cnt = 0; bwtintv_dev ik, ok[4]; bool have = false;
		for (int x = 1; x <= len; ++x) {
			int c = q[x-1];
			if (!have) { d_bwt_set_intv(fm, c, &ik); have = true; }
			else {
				d_bwt_extend(fm, &ik, ok, 0);          /* append c */
				if (ok[3-c].x2 == 0) { ++cnt; d_bwt_set_intv(fm, c, &ik); }
				else ik = ok[3-c];
			}
			bidL[x] = (uint8_t)cnt;
		}
	} else if (lane == 1) {
		bidR[len] = 0;
		int cnt = 0; bwtintv_dev ik, ok[4]; bool have = false;
		for (int x = len - 1; x >= 0; --x) {
			int c = q[x];
			if (!have) { d_bwt_set_intv(fm, c, &ik); have = true; }
			else {
				d_bwt_extend(fm, &ik, ok, 1);          /* prepend c */
				if (ok[c].x2 == 0) { ++cnt; d_bwt_set_intv(fm, c, &ik); }
				else ik = ok[c];
			}
			bidR[x] = (uint8_t)cnt;
		}
	}
}

/* A single search of a search scheme: part order pi, with cumulative error bounds L (lower)
 * and U (upper) checked at part boundaries. pi must be CONTIGUOUS, so the search starts inside
 * part pi[0] and thereafter always extends into an adjacent part -- one base at a time, left or
 * right, which is exactly what a bidirectional index provides.
 *
 * U is applied continuously inside a part (cumulative error never decreases, so enforcing the
 * part's cap early is safe and prunes sooner). L is checked only on COMPLETING a part: it is
 * what stops the searches of a scheme from redoing each other's work, and omitting it makes a
 * scheme look far more expensive than it is. */
struct Search { int P; int pi[8], L[8], U[8]; };

__global__ void k_scheme(fmidx_dev fm, const uint8_t *seq, const uint32_t *off, const int *rlen,
                         int nreads, Search sc, int use_bound, int Kmax,
                         unsigned long long *pops, int *workctr, int *overflow, unsigned long long *hits,
                         uint8_t *rmask)
{
	extern __shared__ unsigned char smem[];
	int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
	BNode *st = (BNode*)smem + (size_t)wib * CAPSM;
	uint8_t *bidbuf = (uint8_t*)((BNode*)smem + (size_t)WPB * CAPSM) + (size_t)wib * 2 * MAXRD;
	uint8_t *bidL = bidbuf, *bidR = bidbuf + MAXRD;
	const unsigned FULL = 0xffffffffu;

	for (;;) {
		int r;
		if (lane == 0) r = atomicAdd(workctr, 1);
		r = __shfl_sync(FULL, r, 0);
		if (r >= nreads) break;

		int len = rlen[r];
		const uint8_t *q = seq + off[r];
		int P = sc.P;
		int pb[9];                                  /* part boundaries */
		for (int t = 0; t <= P; ++t) pb[t] = (int)(((long long)len * t) / P);

		if (use_bound) { d_build_bounds(fm, q, len, bidL, bidR, lane); __syncwarp(); }
		int p0 = sc.pi[0];
		int sp = 0; unsigned long long nn = 0; int ovf = 0, hit = 0;

		/* seed: leftmost base of part pi[0], allowing it to be a mismatch */
		if (lane == 0) {
			int a = pb[p0];
			for (int b = 0; b < 4; ++b) {
				int e = (b != q[a]);
				if (e > sc.U[0]) continue;
				int j = 0, nlo = a, nhi = a + 1;
				if (nhi == pb[p0+1]) { if (e < sc.L[0]) continue; j = 1; }
				bwtintv_dev ik; d_bwt_set_intv(fm, b, &ik);
				st[sp].x0 = ik.x0; st[sp].x1 = ik.x1; st[sp].x2 = ik.x2;
				st[sp].p = bpack(nlo, nhi, e) | ((uint32_t)j << 24); ++sp;
			}
		}
		sp = __shfl_sync(FULL, sp, 0);
		__syncwarp();

		while (sp > 0) {
			int room = CAPSM - sp;
			int n_active = sp < 32 ? sp : 32;
			int r4 = room / 4; if (n_active > r4) n_active = r4;
			if (n_active < 1) { ovf = 1; break; }
			bool active = lane < n_active;

			bwtintv_dev ik; int lo = 0, hi = 0, e = 0, j = 0;
			if (active) {
				BNode nd = st[sp - 1 - lane];
				ik.x0 = nd.x0; ik.x1 = nd.x1; ik.x2 = nd.x2;
				lo = nd.p & 0xff; hi = (nd.p >> 8) & 0xff; e = (nd.p >> 16) & 0x3f; j = (nd.p >> 24) & 0xf;
			}
			sp -= n_active; nn += n_active;

			int is_back = 0, pos = 0, tgt_lo = 0, tgt_hi = 0;
			bool done = false, gen = false;
			if (active) {
				if (j >= P) done = true;               /* whole pattern matched within bounds */
				else {
					int pp = sc.pi[j]; tgt_lo = pb[pp]; tgt_hi = pb[pp+1];
					/* pi is contiguous, so the target part is either to the right of the window
					 * or to the left -- or is the part we are still filling. Extend toward
					 * whichever side of the target is not yet covered. */
					if (tgt_hi > hi)      { is_back = 0; pos = hi; }     /* grow right */
					else if (tgt_lo < lo) { is_back = 1; pos = lo - 1; } /* grow left  */
					else                  { done = true; }              /* part already covered */
					gen = !done && pos >= 0 && pos < len;
				}
			}
			if (__any_sync(FULL, done)) { hit = 1; break; }

			bwtintv_dev ok[4];
			if (gen) d_bwt_extend(fm, &ik, ok, is_back);

			/* CHILD(EMIT): one source of truth for the (pi,L,U) child rules.
			 * U is enforced continuously inside the current part; L only on completing it. */
			#define CHILD(EMIT) do { \
				for (int b = 0; b < 4; ++b) { \
					int ne = e + (b != q[pos]); \
					int ix = is_back ? b : 3 - b;             /* forward extension complements */ \
					if (ne > sc.U[j] || ok[ix].x2 == 0) continue; \
					int nlo = is_back ? lo - 1 : lo, nhi = is_back ? hi : hi + 1; \
					if (use_bound && ne + bidL[nlo] + bidR[nhi] > Kmax) continue; \
					int nj = j; \
					if ((is_back && nlo == tgt_lo) || (!is_back && nhi == tgt_hi)) { \
						if (ne < sc.L[j]) continue;           /* part done but too few errors */ \
						nj = j + 1; \
					} \
					EMIT; \
				} } while (0)

			int nc = 0;
			if (gen) CHILD(++nc);

			int incl = nc;
			for (int d = 1; d < 32; d <<= 1) { int y = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += y; }
			int total = __shfl_sync(FULL, incl, 31);
			if (sp + total > CAPSM) { ovf = 1; break; }
			int wp = sp + incl - nc;
			if (gen) CHILD({ st[wp].x0 = ok[ix].x0; st[wp].x1 = ok[ix].x1; st[wp].x2 = ok[ix].x2;
			                 st[wp].p = bpack(nlo, nhi, ne) | ((uint32_t)nj << 24); ++wp; });
			#undef CHILD

			sp += total;
			__syncwarp();
		}
		if (lane == 0) {
			atomicAdd(pops, nn);
			if (ovf) atomicAdd(overflow, 1);
			if (hit) atomicAdd(hits, 1ULL);
			rmask[r] = (uint8_t)((hit ? 1 : 0) | (ovf ? 2 : 0));   /* per-read hit/overflow */
		}
		__syncwarp();
	}
}

/* minimal plain-FASTQ reader (test sets are uncompressed); skips reads containing N */
static int read_fastq(const char *fn, std::vector<uint8_t> &flat, std::vector<uint32_t> &off,
                      std::vector<int> &len, int maxreads)
{
	FILE *f = fopen(fn, "r");
	if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
	char *line = NULL; size_t cap = 0; ssize_t nl; int nline = 0, n = 0;
	while ((nl = getline(&line, &cap, f)) > 0) {
		if ((nline & 3) == 1) {
			while (nl > 0 && (line[nl-1] == '\n' || line[nl-1] == '\r')) --nl;
			bool ok = nl > 0 && nl < MAXRD;
			uint32_t base = flat.size();
			for (ssize_t i = 0; i < nl && ok; ++i) {
				uint8_t c;
				switch (line[i]) { case 'A': case 'a': c=0; break; case 'C': case 'c': c=1; break;
				                   case 'G': case 'g': c=2; break; case 'T': case 't': c=3; break;
				                   default: ok = false; c=0; }
				flat.push_back(c);
			}
			if (ok) { off.push_back(base); len.push_back((int)nl); if (++n >= maxreads && maxreads>0) break; }
			else flat.resize(base);
		}
		++nline;
	}
	free(line); fclose(f);
	return n;
}

/* Table 3 of Kianfar/Pockrandt/Torkamandi/Luo/Reinert, "Optimum Search Schemes for Approximate
 * String Matching Using Bidirectional FM-Index" (arXiv:1711.02035v2), reproduced verbatim.
 * Each entry is (pi, L, U) with equal-size parts. Covering property verified exhaustively. */
struct SchemeSet { int K, P, ns; const char *tag; const char *pi[8]; const char *L[8]; const char *U[8]; };
static const SchemeSet SCHEMES[] = {
  /* P = 1: plain backtracking (one part, no partitioning) -- the baseline to compare against */
  {1,1,1,"backtrack",{"1"},{"0"},{"1"}},
  {2,1,1,"backtrack",{"1"},{"0"},{"2"}},
  {3,1,1,"backtrack",{"1"},{"0"},{"3"}},
  {4,1,1,"backtrack",{"1"},{"0"},{"4"}},
  /* P = K+1 */
  {1,2,2,"kianfar",{"12","21"},          {"00","01"},          {"01","01"}},
  {2,3,3,"kianfar",{"123","321","231"},  {"002","000","011"},  {"012","022","012"}},
  {3,4,3,"kianfar",{"1234","2341","3421"},{"0003","0000","0022"},{"0233","1223","0033"}},
  {4,5,3,"kianfar",{"12345","23451","54321"},{"00004","00000","00033"},{"03344","22334","00444"}},
  /* P = K+2 */
  {2,4,3,"kianfar",{"2134","3214","4321"},{"0011","0000","0002"},{"0022","0112","0122"}},
  {3,5,3,"kianfar",{"12345","43215","54321"},{"00022","00000","00003"},{"00333","11223","02233"}},
  {4,6,3,"kianfar",{"123456","234561","654321"},{"000004","000000","000033"},{"033344","222334","004444"}},
  /* Qwen literature scan: SeqAn3 k=3 (4 searches) and Renders et al. k=4 (5 searches).
   * Selected with P = 100+p so they do not collide with the Kianfar entries above. */
  {3,105,4,"seqan3",{"54321","34521","23451","12345"},
                    {"00000","00111","00022","00003"},
                    {"00333","01123","01223","02233"}},
  {4,105,5,"renders",{"12345","23145","32145","45321","54321"},
                     {"00222","00000","01111","00003","01114"},
                     {"02244","01244","01244","01444","01444"}},
};

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s <ref.fa> <reads.fq> [K] [P]\n", argv[0]); return 1; }
	int want_K = argc > 3 ? atoi(argv[3]) : 4;
	int want_P = argc > 4 ? atoi(argv[4]) : want_K + 1;
	int use_bound = argc > 5 ? atoi(argv[5]) : 0;

	bwt_t *bwt; { char b[1024]; snprintf(b,sizeof(b),"%s.bwt",argv[1]); bwt = bwt_restore_bwt(b); }
	uint32_t *d_bwt; size_t bsz = (size_t)bwt->bwt_size * 4;
	CK(cudaMalloc(&d_bwt, bsz)); CK(cudaMemcpy(d_bwt, bwt->bwt, bsz, cudaMemcpyHostToDevice));
	CK(cudaMemcpyToSymbol(c_cnt_table, bwt->cnt_table, sizeof(bwt->cnt_table)));
	{ uint64_t L2[5]; for (int i=0;i<5;i++) L2[i]=bwt->L2[i]; CK(cudaMemcpyToSymbol(c_L2, L2, sizeof(L2))); }
	fmidx_dev fm; fm.bwt = d_bwt; fm.primary = bwt->primary; fm.seq_len = bwt->seq_len;

	std::vector<uint8_t> flat; std::vector<uint32_t> off; std::vector<int> len;
	int nreads = read_fastq(argv[2], flat, off, len, 0);

	uint8_t *d_seq; uint32_t *d_off; int *d_len;
	CK(cudaMalloc(&d_seq, flat.size())); CK(cudaMemcpy(d_seq, flat.data(), flat.size(), cudaMemcpyHostToDevice));
	CK(cudaMalloc(&d_off, off.size()*4)); CK(cudaMemcpy(d_off, off.data(), off.size()*4, cudaMemcpyHostToDevice));
	CK(cudaMalloc(&d_len, len.size()*4)); CK(cudaMemcpy(d_len, len.data(), len.size()*4, cudaMemcpyHostToDevice));

	unsigned long long *d_pops, *d_hits; int *d_wc, *d_ovf; uint8_t *d_mask;
	CK(cudaMalloc(&d_pops,8)); CK(cudaMalloc(&d_hits,8)); CK(cudaMalloc(&d_wc,4)); CK(cudaMalloc(&d_ovf,4));
	CK(cudaMalloc(&d_mask, nreads));
	std::vector<uint8_t> acc_hit(nreads,0), acc_ovf(nreads,0), hmask(nreads);

	size_t sh = (size_t)WPB * CAPSM * sizeof(BNode) + (size_t)WPB * 2 * MAXRD;
	CK(cudaFuncSetAttribute(k_scheme, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
	int mb=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_scheme, WPB*32, sh));
	int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
	int nblk = (mb>0?mb:1)*nsm;

	const SchemeSet *ss = NULL;
	for (unsigned i=0;i<sizeof(SCHEMES)/sizeof(SCHEMES[0]);++i)
		if (SCHEMES[i].K==want_K && SCHEMES[i].P==want_P) ss = &SCHEMES[i];
	if (!ss) { fprintf(stderr, "no scheme for K=%d P=%d\n", want_K, want_P); return 1; }

	fprintf(stderr, "[scheme] %d reads, K=%d P=%d (%s), %d searches, two-sided-bound=%d\n",
	        nreads, ss->K, ss->P > 100 ? ss->P - 100 : ss->P, ss->tag, ss->ns, use_bound);
	printf("%-30s %16s %12s %9s %10s\n", "search (pi,L,U)", "node-pops", "pops/read", "overflow", "hits");
	unsigned long long grand = 0, grand_hits = 0;
	for (int t = 0; t < ss->ns; ++t) {
		Search sc; sc.P = (ss->P > 100 ? ss->P - 100 : ss->P);
		for (int x = 0; x < sc.P; ++x) {
			sc.pi[x] = ss->pi[t][x]-'1'; sc.L[x] = ss->L[t][x]-'0'; sc.U[x] = ss->U[t][x]-'0';
		}
		CK(cudaMemset(d_pops,0,8)); CK(cudaMemset(d_hits,0,8));
		CK(cudaMemset(d_wc,0,4)); CK(cudaMemset(d_ovf,0,4));
		CK(cudaMemset(d_mask, 0, nreads));
		k_scheme<<<nblk, WPB*32, sh>>>(fm, d_seq, d_off, d_len, nreads, sc, use_bound, ss->K,
		                               d_pops, d_wc, d_ovf, d_hits, d_mask);
		CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
		unsigned long long P_=0, H_=0; int O=0;
		CK(cudaMemcpy(&P_,d_pops,8,cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(&H_,d_hits,8,cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(&O,d_ovf,4,cudaMemcpyDeviceToHost));
		grand += P_; grand_hits += H_;
		char lbl[64]; snprintf(lbl,sizeof lbl,"(%s,%s,%s)", ss->pi[t], ss->L[t], ss->U[t]);
		printf("%-30s %16llu %12.0f %9d %10llu\n", lbl, P_, (double)P_/nreads, O, H_);
		CK(cudaMemcpy(hmask.data(), d_mask, nreads, cudaMemcpyDeviceToHost));
		for (int i=0;i<nreads;i++){ acc_hit[i] |= (hmask[i]&1); acc_ovf[i] |= ((hmask[i]>>1)&1); }
	}
	printf("%-30s %16llu %12.0f %9s %10llu\n", "SCHEME TOTAL", grand, (double)grand/nreads, "-", grand_hits);
	/* dump the per-read hit set so a scheme can be diffed against plain backtracking:
	 * a covering scheme must find EXACTLY the same reads (modulo stack overflow). */
	if (getenv("BIDIR_DUMP")) {
		FILE *f = fopen(getenv("BIDIR_DUMP"), "wb");
		for (int i=0;i<nreads;i++){ uint8_t v = (uint8_t)(acc_hit[i] | (acc_ovf[i]<<1)); fwrite(&v,1,1,f); }
		fclose(f);
		fprintf(stderr, "[scheme] per-read hit set -> %s\n", getenv("BIDIR_DUMP"));
	}
	bwt_destroy(bwt);
	return 0;
}
