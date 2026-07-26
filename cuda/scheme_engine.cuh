/* Bidirectional search-scheme DFS engine -- part of the bwa `gpualn` GPU port for ancient DNA.
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

/* Replaces the monolithic max_diff-deep backtrack with a covering search scheme run on the
 * bidirectional FM-index (bwa's own .bwt; see fm_device.cuh d_bwt_extend).
 *
 * CONTRACT. This engine only has to produce a SUPERSET of bwa's has_hit -- every read it flags is
 * re-aligned exactly by bwt_match_gap on the CPU (aln_gpu.cu), so false positives cost time, not
 * correctness. False NEGATIVES are the only fatal error. Two things guarantee there are none:
 *   1. the scheme is COVERING (verified exhaustively at startup, schemes.cuh sch_validate);
 *   2. this executor is strictly more permissive than bwa -- it keeps max_gapo/max_gape (bwa's
 *      own alignments obey them) but DROPS indel_end_skip and max_del_occ, which only ever
 *      remove candidates.
 *
 * Part membership is defined by READ position, so an error is attributed to the part holding the
 * read base it consumes (a deletion, which consumes no read base, is attributed to the part being
 * extended). Every edit alignment therefore induces a well-defined error distribution over parts,
 * and a covering scheme admits it -- no boundary slack is required.
 */
#ifndef SCHEME_ENGINE_CUH
#define SCHEME_ENGINE_CUH

#include <stdint.h>
#include "schemes.cuh"

#define SST_M 0
#define SST_I 1
#define SST_D 2
#define SST_E 3        /* EMPTY: no pattern matched yet (seed position not yet chosen) */

#define SCH_MAXLEN 127 /* lo/hi are 7 bits; longer reads fall back to the exact engine */

/* 24-byte node: bidirectional interval + packed state.
 * x2 (interval size) fits u32: the largest single-base interval is ~1.57e9 < 2^32, and intervals
 * only shrink. The full-range interval is never stored (SST_E marks "no pattern yet"). */
struct SNode { uint64_t x0, x1; uint32_t x2, p; };

__device__ __forceinline__ uint32_t spack(int lo,int hi,int mm,int go,int ge,int st,int j)
{ return (uint32_t)lo | ((uint32_t)hi<<7) | ((uint32_t)mm<<14) | ((uint32_t)go<<19)
       | ((uint32_t)ge<<22) | ((uint32_t)st<<26) | ((uint32_t)j<<28); }

/* One search of one scheme for one read, warp-cooperative. Returns 1 if the whole read was
 * matched within the search's bounds (or the budget/stack gave up -> caller flags to CPU). */
__device__ inline int d_scheme_search(const fmidx_dev fm, const uint8_t *seq, int len,
                                      int K, int si, const int *pb,
                                      int max_gapo, int max_gape,
                                      SNode *st, int CAP_SM,
                                      unsigned long long budget, int *flagged,
                                      unsigned long long *nn_io)
{
	const unsigned FULL = 0xffffffffu;
	int lane = threadIdx.x & 31;
	const SchemeTab &T = c_sch[K];
	int P = T.P;
	unsigned long long nn = *nn_io;

	/* seed: EMPTY pattern positioned at the left edge of the first part */
	int sp = 0;
	if (lane == 0) {
		int a = pb[T.pi[si][0]];
		st[0].x0 = 0; st[0].x1 = 0; st[0].x2 = 0;
		st[0].p = spack(a, a, 0, 0, 0, SST_E, 0);
		sp = 1;
	}
	sp = __shfl_sync(FULL, sp, 0);
	__syncwarp();

	while (sp > 0) {
		int room = CAP_SM - sp;
		int n_active = sp < 32 ? sp : 32;
		int r9 = room / 9; if (n_active > r9) n_active = r9;
		if (n_active < 1) { if (lane==0) *flagged = 1; *nn_io = nn; return 1; }
		bool active = lane < n_active;

		uint64_t x0=0, x1=0; uint32_t x2=0;
		int lo=0, hi=0, mm=0, go=0, ge=0, stt=0, j=0;
		if (active) {
			SNode nd = st[sp-1-lane];
			x0=nd.x0; x1=nd.x1; x2=nd.x2; uint32_t q=nd.p;
			lo=q&0x7f; hi=(q>>7)&0x7f; mm=(q>>14)&0x1f; go=(q>>19)&0x7;
			ge=(q>>22)&0xf; stt=(q>>26)&0x3; j=(q>>28)&0x7;
		}
		sp -= n_active; nn += n_active;
		if (nn > budget) { if (lane==0) *flagged = 1; *nn_io = nn; return 1; }

		/* direction: pi is contiguous, so the target part is adjacent to the window */
		bool done=false, gen=false; int is_back=0, pos=0, tgt_lo=0, tgt_hi=0, cap=0, lob=0;
		if (active) {
			if (j >= P) done = true;
			else {
				int pp = T.pi[si][j]; tgt_lo = pb[pp]; tgt_hi = pb[pp+1];
				cap = T.U[si][j]; lob = T.L[si][j];
				if (tgt_hi > hi)      { is_back = 0; pos = hi; }
				else if (tgt_lo < lo) { is_back = 1; pos = lo - 1; }
				else                  { done = true; }        /* zero-length part */
				gen = !done && pos >= 0 && pos < len;
			}
		}
		if (__any_sync(FULL, done)) { *nn_io = nn; return 1; }

		/* one bidirectional extension: 8 counts + the 4 opposite-direction starts, all in
		 * named registers (an addressable array would land in local memory -- see D5). */
		uint64_t t0=0,t1=0,t2=0,t3=0, u0=0,u1=0,u2=0,u3=0, r0=0,r1=0,r2=0,r3=0;
		if (gen && stt != SST_E) {
			uint64_t drive = is_back ? x0 : x1;
			d_bwt_2occ4_s(fm.bwt, fm.primary, drive - 1, drive - 1 + x2,
			              t0,t1,t2,t3, u0,u1,u2,u3);
			uint64_t base = (is_back ? x1 : x0)
			              + (drive <= fm.primary && drive + x2 - 1 >= fm.primary);
			r3 = base;
			r2 = r3 + (u3 - t3);
			r1 = r2 + (u2 - t2);
			r0 = r1 + (u1 - t1);
		}
		#define SNK(b) ((b)==0? t0 : (b)==1? t1 : (b)==2? t2 : t3)
		#define SNL(b) ((b)==0? u0 : (b)==1? u1 : (b)==2? u2 : u3)
		#define SNR(b) ((b)==0? r0 : (b)==1? r1 : (b)==2? r2 : r3)

		/* CH(EMIT) enumerates bwa's child rules under the scheme bounds.
		 * M/I consume a read base (window grows); D consumes a reference base only. */
		#define CH(EMIT) do { \
			int rb = seq[pos]; \
			/* --- match / mismatch (and, from EMPTY, the seed) --- */ \
			for (int b = 0; b < 4; ++b) { \
				int nmm = mm + (b != rb || rb > 3); \
				int ne = nmm + go + ge; \
				if (ne > cap) continue; \
				uint64_t cx0, cx1; uint32_t cx2; \
				if (stt == SST_E) { \
					cx0 = c_L2[b] + 1; cx1 = c_L2[3-b] + 1; \
					cx2 = (uint32_t)(c_L2[b+1] - c_L2[b]); \
				} else { \
					int ix = is_back ? b : 3 - b; \
					uint32_t sz = (uint32_t)(SNL(ix) - SNK(ix)); \
					if (sz == 0) continue; \
					uint64_t fwd = c_L2[ix] + 1 + SNK(ix); \
					cx0 = is_back ? fwd : SNR(ix); cx1 = is_back ? SNR(ix) : fwd; cx2 = sz; \
				} \
				if (cx2 == 0) continue; \
				int nlo = is_back ? lo-1 : lo, nhi = is_back ? hi : hi+1, nj = j; \
				if ((is_back && nlo == tgt_lo) || (!is_back && nhi == tgt_hi)) { \
					if (ne < lob) continue; nj = j + 1; } \
				EMIT(cx0, cx1, cx2, nlo, nhi, nmm, go, ge, SST_M, nj); \
			} \
			/* --- insertion: read base consumed, reference not --- */ \
			if (stt == SST_M || stt == SST_E) { \
				if (go < max_gapo) { int ne = mm + go+1 + ge; \
					if (ne <= cap) { int nlo = is_back ? lo-1 : lo, nhi = is_back ? hi : hi+1, nj = j; \
						int okp = 1; \
						if ((is_back && nlo == tgt_lo) || (!is_back && nhi == tgt_hi)) { \
							if (ne < lob) okp = 0; else nj = j + 1; } \
						if (okp) EMIT(x0, x1, x2, nlo, nhi, mm, go+1, ge, \
						              stt == SST_E ? SST_E : SST_I, nj); } } \
			} else if (stt == SST_I) { \
				if (ge < max_gape) { int ne = mm + go + ge+1; \
					if (ne <= cap) { int nlo = is_back ? lo-1 : lo, nhi = is_back ? hi : hi+1, nj = j; \
						int okp = 1; \
						if ((is_back && nlo == tgt_lo) || (!is_back && nhi == tgt_hi)) { \
							if (ne < lob) okp = 0; else nj = j + 1; } \
						if (okp) EMIT(x0, x1, x2, nlo, nhi, mm, go, ge+1, SST_I, nj); } } \
			} \
			/* --- deletion: reference base consumed, read not (window unchanged) --- */ \
			if (stt != SST_E) { \
				int opening = (stt == SST_M); \
				int allow = opening ? (go < max_gapo) : (stt == SST_D && ge < max_gape); \
				if (allow) { \
					int ngo = go + (opening ? 1 : 0), nge = ge + (opening ? 0 : 1); \
					int ne = mm + ngo + nge; \
					if (ne <= cap) for (int b = 0; b < 4; ++b) { \
						int ix = is_back ? b : 3 - b; \
						uint32_t sz = (uint32_t)(SNL(ix) - SNK(ix)); \
						if (sz == 0) continue; \
						uint64_t fwd = c_L2[ix] + 1 + SNK(ix); \
						uint64_t cx0 = is_back ? fwd : SNR(ix), cx1 = is_back ? SNR(ix) : fwd; \
						EMIT(cx0, cx1, sz, lo, hi, mm, ngo, nge, SST_D, j); \
					} \
				} \
			} \
		} while (0)

		int nc = 0;
		#define CNT(a,b,c,d,e,f,g,h,i,k) (void)(++nc)
		if (gen) CH(CNT);
		#undef CNT

		int incl = nc;
		for (int d=1; d<32; d<<=1) { int y = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += y; }
		int total = __shfl_sync(FULL, incl, 31);
		if (sp + total > CAP_SM) { if (lane==0) *flagged = 1; *nn_io = nn; return 1; }
		int wp = sp + incl - nc;

		#define PUT(cx0,cx1,cx2,nlo,nhi,nmm,ngo,nge,nst,nj) do { \
			st[wp].x0=(cx0); st[wp].x1=(cx1); st[wp].x2=(cx2); \
			st[wp].p=spack((nlo),(nhi),(nmm),(ngo),(nge),(nst),(nj)); ++wp; } while(0)
		if (gen) CH(PUT);
		#undef PUT
		#undef CH
		#undef SNK
		#undef SNL
		#undef SNR

		sp += total;
		__syncwarp();
	}
	*nn_io = nn;
	return 0;
}

/* Run every search of the scheme for one read; has_hit is their disjunction, so the first hit
 * ends the read. */
__device__ inline int d_scheme_has_hit(const fmidx_dev fm, const uint8_t *seq, int len, int K,
                                       int max_gapo, int max_gape, SNode *st, int CAP_SM,
                                       unsigned long long budget, int *flagged,
                                       unsigned long long *nn_out)
{
	int lane = threadIdx.x & 31;
	int nN = 0;
	for (int j = 0; j < len; ++j) if (seq[j] > 3) ++nN;
	if (nN > K) { if (lane==0) *nn_out = 0; return 0; }

	const SchemeTab &T = c_sch[K];
	int pb[SCH_MAXP + 1];
	for (int t = 0; t <= T.P; ++t) pb[t] = (int)(((long long)len * t) / T.P);

	unsigned long long nn = 0;
	for (int si = 0; si < T.ns; ++si) {
		int fl = 0;
		int h = d_scheme_search(fm, seq, len, K, si, pb, max_gapo, max_gape,
		                        st, CAP_SM, budget, &fl, &nn);
		fl = __shfl_sync(0xffffffffu, fl, 0);
		if (fl) { if (lane==0) { *flagged = 1; *nn_out = nn; } return 1; }
		if (h)  { if (lane==0) *nn_out = nn; return 1; }
	}
	if (lane==0) *nn_out = nn;
	return 0;
}

/* Persistent work-pool kernel. Include AFTER dfs_engine.cuh: reads whose max_diff has no
 * validated scheme (K>=5) or that exceed SCH_MAXLEN fall back to the exact backtracking engine,
 * so correctness never depends on the scheme table's coverage of K.
 * For the production short branch (L<=63 => K=3,4) the fallback is never taken. */
__global__ void k_dfs_scheme(fmidx_dev fm, const uint8_t *seq, const uint64_t *w_w, const int *w_bid,
                             const ReadParam *rp, int nreads, int max_gapo, int max_gape, int mode,
                             int indel_end_skip, int max_del_occ, int CAP_SM, int CAP_GL,
                             uint64_t *Gk, uint64_t *Gl, uint32_t *Gn, uint8_t *has_hit,
                             int *workctr, unsigned long long *npop, unsigned long long budget,
                             int *nflag, int wpb, uint8_t *flag_out, const int *order,
                             int *nfallback)
{
	extern __shared__ unsigned char smem[];
	int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
	SNode *sst = (SNode*)smem + (size_t)wib * CAP_SM;
	/* the fallback engine's SoA stack is carved from the same 24 B/entry allocation (20 B fits) */
	uint64_t *sk = (uint64_t*)sst;
	uint64_t *sl = sk + CAP_SM;
	uint32_t *sn = (uint32_t*)(sl + CAP_SM);
	int gslot = blockIdx.x * wpb + wib;
	uint64_t *gk = Gk + (size_t)gslot * CAP_GL;
	uint64_t *gl = Gl + (size_t)gslot * CAP_GL;
	uint32_t *gn = Gn + (size_t)gslot * CAP_GL;

	for (;;) {
		int idx;
		if (lane == 0) idx = atomicAdd(workctr, 1);
		idx = __shfl_sync(0xffffffffu, idx, 0);
		if (idx >= nreads) break;
		int r = order ? order[idx] : idx;
		ReadParam p = rp[r];

		int flagged = 0; unsigned long long nn = 0; int hh;
		if (p.max_diff >= 1 && p.max_diff <= SCH_MAXK && p.len <= SCH_MAXLEN) {
			hh = d_scheme_has_hit(fm, seq + p.seq_off, p.len, p.max_diff,
			                      max_gapo, max_gape, sst, CAP_SM, budget, &flagged, &nn);
		} else {
			if (lane == 0) atomicAdd(nfallback, 1);
			hh = d_dfs_has_hit_warp2(fm, seq + p.seq_off, p.len, w_w + p.w_off, w_bid + p.w_off,
				p.max_diff, max_gapo, max_gape, mode, indel_end_skip, max_del_occ,
				sk, sl, sn, CAP_SM, gk, gl, gn, CAP_GL, budget, &flagged, &nn INSTR_PASS_ZERO);
		}
		if (lane == 0) { has_hit[r] = (uint8_t)hh; npop[r] = nn;
			if (flag_out) flag_out[r] = (uint8_t)flagged; if (flagged) atomicAdd(nflag, 1); }
	}
}

#endif /* SCHEME_ENGINE_CUH */
