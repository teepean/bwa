/* Dual-read warp engine with 12-byte nodes: genuine per-lane MLP=2 at full occupancy.
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

/* WHY 12 BYTES. Phase 17 measured that per-lane MLP 1->2 triples random-gather throughput on
 * sm_86 (2,375 -> 7,355 M-probe/s) -- but only when the WARP COUNT is held fixed, i.e. going from
 * 256 to 512 outstanding requests per SM. Phase 18's first dual-read attempt used 20-byte nodes,
 * so two stacks per warp halved occupancy to 4 warps/SM, giving 4x26x2 = 208 outstanding -- the
 * SAME total as 8x26x1. Identical in-flight, minus warp-level latency hiding: it lost.
 *
 * Reaching the measured MLP=2 point needs 8 warps/SM AND two reads per warp = 512 outstanding,
 * which means two stacks inside the shared-memory budget of one. Hence the node shrinks:
 *
 *   k, l < seq_len (6.27e9) < 2^33  =>  33 + 33 bits
 *   i(9) n_mm(6) n_gapo(4) n_gape(4) state(2)  =>  25 bits
 *   total 91 bits, stored as 3 x uint32 = 12 bytes (was 20).
 *
 * 2 stacks x 512 entries x 12 B = 12 KB/warp -> 48 KB/block at wpb=4 -> 2 blocks/SM -> 8 warps/SM.
 *
 * The 33-bit assumption is genome-dependent: seq_len = 2 x reference length, so it holds up to a
 * ~4.29 Gbp reference. The host MUST check it (see n12_fits) and fall back to the 20-byte engine
 * otherwise -- silently truncating k or l would corrupt the search.
 *
 * Both slots keep the two-level shared+global stack of k_dfs_warp2: without it, overflow flags
 * ~3.5% of reads to the CPU instead of ~0.5%, which costs far more than the kernel saves.
 */
#ifndef DUAL_ENGINE_CUH
#define DUAL_ENGINE_CUH

#include <stdint.h>

#define N12_MAX_SEQLEN (1ULL << 33)
static inline int n12_fits(uint64_t seq_len) { return seq_len < N12_MAX_SEQLEN; }

__device__ __forceinline__ void n12_pack(uint64_t k, uint64_t l, int i, int mm, int go, int ge, int st,
                                         uint32_t &A, uint32_t &B, uint32_t &C)
{
	A = (uint32_t)k;
	B = (uint32_t)l;
	C = (uint32_t)(k >> 32) | ((uint32_t)(l >> 32) << 1) | ((uint32_t)i << 2)
	  | ((uint32_t)mm << 11) | ((uint32_t)go << 17) | ((uint32_t)ge << 21) | ((uint32_t)st << 25);
}
__device__ __forceinline__ void n12_unpack(uint32_t A, uint32_t B, uint32_t C,
                                           uint64_t &k, uint64_t &l, int &i, int &mm, int &go, int &ge, int &st)
{
	k  = (uint64_t)A | ((uint64_t)(C & 1u) << 32);
	l  = (uint64_t)B | ((uint64_t)((C >> 1) & 1u) << 32);
	i  = (C >> 2) & 0x1ff; mm = (C >> 11) & 0x3f;
	go = (C >> 17) & 0xf;  ge = (C >> 21) & 0xf; st = (C >> 25) & 0x3;
}

/* per-slot register state; named registers only (an addressable array lands in local memory) */
#define DUAL_DECL(S) \
	bool actv##S = false, gen##S = false, hit##S = false; \
	uint64_t k##S=0, l##S=0, occ##S=0; \
	uint64_t ca0##S=0,ca1##S=0,ca2##S=0,ca3##S=0,cb0##S=0,cb1##S=0,cb2##S=0,cb3##S=0; \
	int i##S=0, e_mm##S=0, e_go##S=0, e_ge##S=0, e_st##S=0; \
	int xi##S=0, allow_diff##S=1, allow_M##S=1, tmp##S=0, m##S=0, nc##S=0;

#define DUAL_SELK(S,x) ((x)==0? ca0##S : (x)==1? ca1##S : (x)==2? ca2##S : ca3##S)
#define DUAL_SELL(S,x) ((x)==0? cb0##S : (x)==1? cb1##S : (x)==2? cb2##S : cb3##S)

/* bwa's child rules, one source of truth; ADD2 is redefined per pass */
#define DUAL_GEN(S, seqp, lenv, mdv) do { \
	if (allow_diff##S && xi##S >= indel_end_skip+tmp##S && lenv-xi##S >= indel_end_skip+tmp##S) { \
		if (e_st##S == STATE_M) { \
			if (e_go##S < max_gapo) { \
				ADD2(xi##S, k##S, l##S, e_mm##S, e_go##S+1, e_ge##S, STATE_I); \
				_Pragma("unroll") for (int j=0;j<4;++j){ uint64_t dk=c_L2[j]+DUAL_SELK(S,j)+1, dl=c_L2[j]+DUAL_SELL(S,j); \
					if (dk<=dl) ADD2(xi##S+1, dk, dl, e_mm##S, e_go##S+1, e_ge##S, STATE_D); } \
			} \
		} else if (e_st##S == STATE_I) { \
			if (e_ge##S < max_gape) ADD2(xi##S, k##S, l##S, e_mm##S, e_go##S, e_ge##S+1, STATE_I); \
		} else { \
			if (e_ge##S < max_gape && (e_ge##S+e_go##S < mdv || occ##S < (uint64_t)max_del_occ)) \
				_Pragma("unroll") for (int j=0;j<4;++j){ uint64_t dk=c_L2[j]+DUAL_SELK(S,j)+1, dl=c_L2[j]+DUAL_SELL(S,j); \
					if (dk<=dl) ADD2(xi##S+1, dk, dl, e_mm##S, e_go##S, e_ge##S+1, STATE_D); } \
		} \
	} \
	if (allow_diff##S && allow_M##S) { \
		_Pragma("unroll") for (int j=1;j<=4;++j){ int c=(seqp[xi##S]+j)&3; int is_mm=(j!=4||seqp[xi##S]>3); \
			uint64_t mk=c_L2[c]+DUAL_SELK(S,c)+1, ml=c_L2[c]+DUAL_SELL(S,c); \
			if (mk<=ml) ADD2(xi##S, mk, ml, e_mm##S+is_mm, e_go##S, e_ge##S, STATE_M); } \
	} else if (seqp[xi##S] < 4) { \
		int c=seqp[xi##S]&3; uint64_t mk=c_L2[c]+DUAL_SELK(S,c)+1, ml=c_L2[c]+DUAL_SELL(S,c); \
		if (mk<=ml) ADD2(xi##S, mk, ml, e_mm##S, e_go##S, e_ge##S, STATE_M); \
	} \
} while (0)

/* two-level stack maintenance for one slot: unspill when empty, spill when nearly full */
#define DUAL_STACK(S, A, B, C, gA, gB, gC, sp, gsp) do { \
	if (sp == 0) { \
		if (gsp == 0) { done##S = 1; break; } \
		int cN = gsp < 128 ? gsp : 128; \
		for (int j = lane; j < cN; j += 32) { A[j]=gA[gsp-cN+j]; B[j]=gB[gsp-cN+j]; C[j]=gC[gsp-cN+j]; } \
		gsp -= cN; sp = cN; __syncwarp(); \
	} \
	if (CAP_SM - sp < 9) { \
		if (gsp + 128 > CAP_GL) { flag##S = 1; done##S = 1; break; } \
		for (int j = lane; j < 128; j += 32) { gA[gsp+j]=A[j]; gB[gsp+j]=B[j]; gC[gsp+j]=C[j]; } \
		__syncwarp(); \
		for (int j = lane; j < sp - 128; j += 32) { A[j]=A[j+128]; B[j]=B[j+128]; C[j]=C[j+128]; } \
		gsp += 128; sp -= 128; __syncwarp(); \
	} \
} while (0)

/* pop one wave for slot S and ISSUE its probe -- deliberately does NOT consume the ca/cb counts */
#define DUAL_POP(S, A, B, C, sp, nn, seqp, lenv, mdv, wbid, ww) do { \
	int room = CAP_SM - sp; \
	int na = sp < 32 ? sp : 32; \
	int r9 = room / 9; if (na > r9) na = r9; \
	if (na < 1) { flag##S = 1; done##S = 1; break; } \
	actv##S = lane < na; \
	if (actv##S) { int ix = sp-1-lane; \
		n12_unpack(A[ix], B[ix], C[ix], k##S, l##S, i##S, e_mm##S, e_go##S, e_ge##S, e_st##S); } \
	sp -= na; nn += na; \
	if (nn > budget) { flag##S = 1; done##S = 1; break; } \
	if (actv##S) { \
		m##S = mdv - (e_mm##S + e_go##S); \
		if (mode & BWA_MODE_GAPE) m##S -= e_ge##S; \
		bool prune = (m##S < 0) || (i##S > 0 && m##S < wbid[i##S-1]); \
		if (!prune) { \
			if (i##S == 0) hit##S = true; \
			else if (m##S == 0 && (e_st##S==STATE_M || (mode&BWA_MODE_GAPE) || e_ge##S==max_gape)) { \
				uint64_t kk=k##S, ll=l##S; \
				if (d_bwt_match_exact_alt_s(fm, seqp, i##S, &kk, &ll)) hit##S = true; else prune = true; \
			} \
			if (!hit##S && !prune) { \
				xi##S = i##S - 1; \
				d_bwt_2occ4_s(fm.bwt, fm.primary, k##S - 1, l##S, \
				              ca0##S,ca1##S,ca2##S,ca3##S, cb0##S,cb1##S,cb2##S,cb3##S); \
				occ##S = l##S - k##S + 1; \
				if (xi##S > 0) { \
					if (wbid[xi##S-1] > m##S-1) allow_diff##S = 0; \
					else if (wbid[xi##S-1]==m##S-1 && wbid[xi##S]==m##S-1 && ww[xi##S-1]==ww[xi##S]) allow_M##S = 0; \
				} \
				tmp##S = (mode & BWA_MODE_LOGGAP) ? d_int_log2(e_ge##S+e_go##S)/2+1 : e_go##S+e_ge##S; \
				gen##S = true; \
			} \
		} \
	} \
} while (0)

/* count -> warp prefix-sum -> emit into slot S's own stack */

__global__ void k_dfs_warp2_dual(fmidx_dev fm, const uint8_t *seq, const uint64_t *w_w, const int *w_bid,
                                 const ReadParam *rp, int nreads, int max_gapo, int max_gape, int mode,
                                 int indel_end_skip, int max_del_occ, int CAP_SM, int CAP_GL,
                                 uint32_t *G, uint8_t *has_hit, int *workctr, unsigned long long *npop,
                                 unsigned long long budget, int *nflag, int wpb,
                                 uint8_t *flag_out, const int *order)
{
	extern __shared__ unsigned char smem[];
	int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const unsigned FULL = 0xffffffffu;
	uint32_t *S32 = (uint32_t*)smem;
	size_t plane = (size_t)wpb * 2 * CAP_SM;          /* one plane per component, all warps/slots */
	uint32_t *Aa = S32 + (size_t)(wib*2+0)*CAP_SM,          *Ab = S32 + (size_t)(wib*2+1)*CAP_SM;
	uint32_t *Ba = S32 + plane   + (size_t)(wib*2+0)*CAP_SM, *Bb = S32 + plane   + (size_t)(wib*2+1)*CAP_SM;
	uint32_t *Ca = S32 + 2*plane + (size_t)(wib*2+0)*CAP_SM, *Cb = S32 + 2*plane + (size_t)(wib*2+1)*CAP_SM;

	size_t gslot = (size_t)(blockIdx.x * wpb + wib) * 2;
	uint32_t *gAa = G + (gslot+0)*CAP_GL*3, *gBa = gAa + CAP_GL, *gCa = gBa + CAP_GL;
	uint32_t *gAb = G + (gslot+1)*CAP_GL*3, *gBb = gAb + CAP_GL, *gCb = gBb + CAP_GL;

	for (;;) {
		int base;
		if (lane == 0) base = atomicAdd(workctr, 2);
		base = __shfl_sync(FULL, base, 0);
		if (base >= nreads) break;
		int iA = base, iB = base + 1;
		int rA = order ? order[iA] : iA;
		int rB = (iB < nreads) ? (order ? order[iB] : iB) : -1;

		ReadParam pA = rp[rA];
		ReadParam pB = (rB >= 0) ? rp[rB] : pA;
		const uint8_t *seqA = seq + pA.seq_off,  *seqB = seq + pB.seq_off;
		const uint64_t *wwA = w_w + pA.w_off,    *wwB = w_w + pB.w_off;
		const int *wbA = w_bid + pA.w_off,       *wbB = w_bid + pB.w_off;
		int lenA = pA.len, lenB = pB.len, mdA = pA.max_diff, mdB = pB.max_diff;

		int spA = 0, spB = 0, gspA = 0, gspB = 0;
		int doneA = 0, doneB = (rB < 0), flagA = 0, flagB = 0, resA = 0, resB = 0;
		unsigned long long nnA = 0, nnB = 0;

		{ int nN=0; for (int j=0;j<lenA;++j) if (seqA[j]>3) ++nN;
		  if (nN > mdA) doneA = 1;
		  else { if (lane==0) n12_pack(0, fm.seq_len, lenA,0,0,0,STATE_M, Aa[0],Ba[0],Ca[0]); spA=1; } }
		if (!doneB) { int nN=0; for (int j=0;j<lenB;++j) if (seqB[j]>3) ++nN;
		  if (nN > mdB) doneB = 1;
		  else { if (lane==0) n12_pack(0, fm.seq_len, lenB,0,0,0,STATE_M, Ab[0],Bb[0],Cb[0]); spB=1; } }
		__syncwarp();

		while (!doneA || !doneB) {
			DUAL_DECL(A) DUAL_DECL(B)
			int wpA = 0, wpB = 0;

			if (!doneA) DUAL_STACK(A, Aa, Ba, Ca, gAa, gBa, gCa, spA, gspA);
			if (!doneB) DUAL_STACK(B, Ab, Bb, Cb, gAb, gBb, gCb, spB, gspB);

			/* both probes issued before either result is consumed -- this is the MLP=2 */
			if (!doneA) DUAL_POP(A, Aa, Ba, Ca, spA, nnA, seqA, lenA, mdA, wbA, wwA);
			if (!doneB) DUAL_POP(B, Ab, Bb, Cb, spB, nnB, seqB, lenB, mdB, wbB, wwB);

			if (!doneA && __any_sync(FULL, hitA)) { resA = 1; doneA = 1; }
			if (!doneB && __any_sync(FULL, hitB)) { resB = 1; doneB = 1; }

			if (!doneA) {
				#define ADD2(_i,_k,_l,_mm,_go,_ge,_st) do { ++ncA; } while(0)
				ncA = 0; if (genA) DUAL_GEN(A, seqA, lenA, mdA);
				#undef ADD2
				int incl = ncA;
				for (int d=1; d<32; d<<=1) { int y = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += y; }
				int total = __shfl_sync(FULL, incl, 31);
				wpA = spA + incl - ncA;
				#define ADD2(_i,_k,_l,_mm,_go,_ge,_st) do { \
					n12_pack((_k),(_l),(_i),(_mm),(_go),(_ge),(_st), Aa[wpA],Ba[wpA],Ca[wpA]); ++wpA; } while(0)
				if (genA) DUAL_GEN(A, seqA, lenA, mdA);
				#undef ADD2
				spA += total;
			}
			if (!doneB) {
				#define ADD2(_i,_k,_l,_mm,_go,_ge,_st) do { ++ncB; } while(0)
				ncB = 0; if (genB) DUAL_GEN(B, seqB, lenB, mdB);
				#undef ADD2
				int incl = ncB;
				for (int d=1; d<32; d<<=1) { int y = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += y; }
				int total = __shfl_sync(FULL, incl, 31);
				wpB = spB + incl - ncB;
				#define ADD2(_i,_k,_l,_mm,_go,_ge,_st) do { \
					n12_pack((_k),(_l),(_i),(_mm),(_go),(_ge),(_st), Ab[wpB],Bb[wpB],Cb[wpB]); ++wpB; } while(0)
				if (genB) DUAL_GEN(B, seqB, lenB, mdB);
				#undef ADD2
				spB += total;
			}
			__syncwarp();
		}

		if (lane == 0) {
			has_hit[rA] = (uint8_t)(resA | flagA); npop[rA] = nnA;
			if (flag_out) flag_out[rA] = (uint8_t)flagA; if (flagA) atomicAdd(nflag, 1);
			if (rB >= 0) { has_hit[rB] = (uint8_t)(resB | flagB); npop[rB] = nnB;
				if (flag_out) flag_out[rB] = (uint8_t)flagB; if (flagB) atomicAdd(nflag, 1); }
		}
		__syncwarp();
	}
}

#endif /* DUAL_ENGINE_CUH */
