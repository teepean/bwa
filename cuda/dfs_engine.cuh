/* Warp-cooperative two-level-stack DFS engine -- part of the bwa `gpualn` GPU
   BWA-backtrack port for ancient DNA.
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

/* Shared warp-cooperative two-level-stack DFS engine (the production engine).
 * Include AFTER fm_device.cuh. Used by both cuda/dfstest.cu (validation) and
 * cuda/aln_gpu.cu (the streaming bwa-aln tool). See cuda/PROGRESS.md.
 */
#ifndef DFS_ENGINE_CUH
#define DFS_ENGINE_CUH

#include <stdint.h>

#ifndef DFS_STATE_DEFS
#define DFS_STATE_DEFS
#define STATE_M 0
#define STATE_I 1
#define STATE_D 2
#endif

/* per-read parameters that vary (other opts are kernel scalars, identical for all reads) */
struct ReadParam { uint32_t seq_off, w_off; int len, max_diff; };

/* ---- opt-in instrumentation (build with -DDFS_INSTRUMENT; production codegen unaffected) ----
 * Counts FM-index probes and buckets so the kernel's occ4 rate can be compared against the
 * measured 2.32 G-occ4/s random-gather ceiling, plus a (depth, errors) histogram of node pops
 * -- sampled 1-in-DFS_INSTR_MOD reads -- which is what sizes a search-scheme redesign. */
#ifdef DFS_INSTRUMENT
#define DHIST_D 96
#define DHIST_E 8
#define DFS_INSTR_MOD 128
__device__ unsigned long long g_probes  = 0;   /* d_bwt_2occ4 calls (lane-granular)  */
__device__ unsigned long long g_buckets = 0;   /* 64 B buckets those calls touch     */
__device__ unsigned long long g_pops    = 0;   /* node pops (lane-granular)          */
__device__ unsigned long long g_waves   = 0;   /* expand waves (pops/waves = mean active lanes) */
__device__ unsigned long long g_spills  = 0;   /* shared->global spill events        */
__device__ unsigned long long g_dhist[DHIST_D * DHIST_E];
#define INSTR_DECL   unsigned long long ip = 0, ib = 0, ipop = 0, iw = 0, isp = 0;
#define INSTR_WAVE(a)  do { if (a) ++iw; } while (0)
#define INSTR_SPILL(a) do { if (a) ++isp; } while (0)
#define INSTR_ARGS   , int instr
#define INSTR_PASS   , instr
#define INSTR_PASS_ZERO , 0
#define INSTR_FLUSH() do { \
	if (ipop) atomicAdd(&g_pops, ipop); \
	if (ip)   atomicAdd(&g_probes, ip); \
	if (ib)   atomicAdd(&g_buckets, ib); \
	if (iw)   atomicAdd(&g_waves, iw); \
	if (isp)  atomicAdd(&g_spills, isp); } while (0)
#define INSTR_POP(a)          do { if (a) ++ipop; } while (0)
#define INSTR_PROBE(pr,ka,la) do { ++ip; ib += d_2occ4_buckets((pr),(ka),(la)); } while (0)
#define INSTR_HIST(on,d,e)    do { if (on) { int _d = (d), _e = (e); \
	if (_e >= DHIST_E) _e = DHIST_E - 1; \
	if (_d >= 0 && _d < DHIST_D && _e >= 0) atomicAdd(&g_dhist[_d * DHIST_E + _e], 1ULL); } } while (0)
#define MATCH_EXACT_ALT(f,s,n,pk,pl) d_bwt_match_exact_alt_c((f),(s),(n),(pk),(pl), ip, ib)
/* Search-scheme cost probe (Idea A). Emulates ONE search of a p-part scheme with a staircase
 * error budget U: a child at read-depth d = len - i may carry at most U(d) errors. Children
 * exceeding it are never pushed, so g_pops measures that search's TRUE tree size.
 * NOTE: a single staircase search is NOT a superset on its own -- has_hit/.sai from a run with
 * g_stair_on are MEANINGLESS. This is a cost measurement only. */
__device__ int g_stair_on = 0, g_stair_p = 3, g_stair_u1 = 1, g_stair_u2 = 2;
__device__ __forceinline__ int d_stair_cap(int depth, int len, int max_diff)
{
	int b1 = len / g_stair_p, b2 = (2 * len) / g_stair_p;
	if (depth <= b1) return g_stair_u1;
	if (depth <= b2) return g_stair_u2;
	return max_diff;
}
#define STAIR_OK(_i,_mm,_go,_ge) (!g_stair_on || \
	((_mm)+(_go)+(_ge)) <= d_stair_cap(len-(_i), len, max_diff))
#else
#define STAIR_OK(_i,_mm,_go,_ge) 1
#define INSTR_DECL
#define INSTR_ARGS
#define INSTR_PASS
#define INSTR_PASS_ZERO
#define INSTR_FLUSH() do {} while (0)
#define INSTR_POP(a)          do {} while (0)
#define INSTR_PROBE(pr,ka,la) do {} while (0)
#define INSTR_HIST(on,d,e)    do {} while (0)
#define INSTR_WAVE(a)         do {} while (0)
#define INSTR_SPILL(a)        do {} while (0)
#define MATCH_EXACT_ALT(f,s,n,pk,pl) d_bwt_match_exact_alt_s((f),(s),(n),(pk),(pl))
#endif

/* DFS node packed into a u32: i(0-8) | n_mm(9-14) | n_gapo(15-18) | n_gape(19-22) | state(23-24) */
__device__ __forceinline__ uint32_t pack_node(int i, int mm, int go, int ge, int st)
{ return (uint32_t)i | (mm<<9) | (go<<15) | (ge<<19) | (st<<23); }

/* Two-level stack: per-warp shared top-window (CAP_SM) + per-warp GLOBAL backing.
 * Detects has_hit (existence) for a read; bushy reads spill their deep frontier to global and
 * stay on the GPU; only the 2M budget (or frontier > CAP_GL) flags to the CPU. has_hit-only ->
 * traversal order/shape is irrelevant -> bit-exact when paired with exact CPU reconcile. */
__device__ inline int d_dfs_has_hit_warp2(const fmidx_dev fm, const uint8_t *seq, int len,
                                   const uint64_t *w_w, const int *w_bid, int max_diff,
                                   int max_gapo, int max_gape, int mode, int indel_end_skip,
                                   int max_del_occ, uint64_t *sk, uint64_t *sl, uint32_t *sn, int CAP_SM,
                                   uint64_t *gk, uint64_t *gl, uint32_t *gn, int CAP_GL,
                                   unsigned long long budget, int *flagged, unsigned long long *nn_out
                                   INSTR_ARGS)
{
	const unsigned FULL = 0xffffffffu; const int CHUNK = 128;
	int lane = threadIdx.x & 31;
	INSTR_DECL
	int nN = 0;
	for (int j = 0; j < len; ++j) if (seq[j] > 3) ++nN;
	if (nN > max_diff) { if (lane==0) *nn_out = 0; return 0; }

	if (lane == 0) { sk[0]=0; sl[0]=fm.seq_len; sn[0]=pack_node(len,0,0,0,STATE_M); }
	int sp = 1, gsp = 0;
	unsigned long long nn = 0;
	__syncwarp();

	for (;;) {
		if (sp == 0) {
			if (gsp == 0) { if (lane==0) *nn_out = nn; INSTR_FLUSH(); return 0; }
			int c = gsp < CHUNK ? gsp : CHUNK;
			for (int j = lane; j < c; j += 32) { sk[j]=gk[gsp-c+j]; sl[j]=gl[gsp-c+j]; sn[j]=gn[gsp-c+j]; }
			gsp -= c; sp = c; __syncwarp();
		}
		int room = CAP_SM - sp;
		if (room < 9) {
			if (gsp + CHUNK > CAP_GL) { if (lane==0){ *flagged=1; *nn_out=nn; } INSTR_FLUSH(); return 1; }
			for (int j = lane; j < CHUNK; j += 32) { gk[gsp+j]=sk[j]; gl[gsp+j]=sl[j]; gn[gsp+j]=sn[j]; }
			__syncwarp();
			for (int j = lane; j < sp - CHUNK; j += 32) { sk[j]=sk[j+CHUNK]; sl[j]=sl[j+CHUNK]; sn[j]=sn[j+CHUNK]; }
			gsp += CHUNK; sp -= CHUNK; __syncwarp();
			room = CAP_SM - sp;
			INSTR_SPILL(lane == 0);
		}
		int n_active = sp < 32 ? sp : 32;
		int r9 = room / 9; if (n_active > r9) n_active = r9; if (n_active < 1) n_active = 1;
		bool active = lane < n_active;
		uint64_t k=0, l=0; int i=0, e_mm=0, e_go=0, e_ge=0, e_st=0;
		if (active) { int idx = sp-1-lane; k=sk[idx]; l=sl[idx]; uint32_t nd=sn[idx];
			i=nd&0x1ff; e_mm=(nd>>9)&0x3f; e_go=(nd>>15)&0xf; e_ge=(nd>>19)&0xf; e_st=(nd>>23)&0x3; }
		sp -= n_active; nn += n_active;
		INSTR_POP(active);
		INSTR_WAVE(lane == 0);
		INSTR_HIST(instr && active, len - i, e_mm + e_go + e_ge);
		if (nn > budget) { if (lane==0){ *flagged=1; *nn_out=nn; } INSTR_FLUSH(); return 1; }

		/* Child generation runs TWICE over the same macro body: pass 1 only counts (to feed the
		 * warp prefix-sum), pass 2 writes straight into the shared stack at the reserved offset.
		 * Children are a pure function of (cntk,cntl,xi,...) which stay in registers, so
		 * regenerating them costs a few ALU ops -- far cheaper than the 180 B/thread local-memory
		 * child buffer this replaces. Identical emission order, so the visited node set is
		 * unchanged and the engine stays bit-exact. */
		bool hit = false, gen = false;
		int nc = 0, xi = 0, allow_diff = 1, allow_M = 1, tmp = 0, m = 0;
		/* the four Occ counts live in named registers; indexing an array by the RUNTIME base
		 * `c` below would make it address-taken and force it back into local memory */
		uint64_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0, occ = 0;
		#define SELK(x) ((x)==0? a0 : (x)==1? a1 : (x)==2? a2 : a3)
		#define SELL(x) ((x)==0? b0 : (x)==1? b1 : (x)==2? b2 : b3)
		if (active) {
			m = max_diff - (e_mm + e_go);
			if (mode & BWA_MODE_GAPE) m -= e_ge;
			bool prune = (m < 0) || (i > 0 && m < w_bid[i-1]);
			if (!prune) {
				if (i == 0) hit = true;
				else if (m == 0 && (e_st==STATE_M || (mode&BWA_MODE_GAPE) || e_ge==max_gape)) {
					uint64_t kk=k, ll=l;
					if (MATCH_EXACT_ALT(fm, seq, i, &kk, &ll)) hit = true; else prune = true;
				}
				if (!hit && !prune) {
					xi = i - 1;
					INSTR_PROBE(fm.primary, k - 1, l);
					d_bwt_2occ4_s(fm.bwt, fm.primary, k - 1, l, a0,a1,a2,a3, b0,b1,b2,b3);
					occ = l - k + 1;
					if (xi > 0) {
						if (w_bid[xi-1] > m-1) allow_diff = 0;
						else if (w_bid[xi-1]==m-1 && w_bid[xi]==m-1 && w_w[xi-1]==w_w[xi]) allow_M = 0;
					}
					tmp = (mode & BWA_MODE_LOGGAP) ? d_int_log2(e_ge+e_go)/2+1 : e_go+e_ge;
					gen = true;
				}
			}
		}
		if (__any_sync(FULL, hit)) { if (lane==0) *nn_out = nn; INSTR_FLUSH(); return 1; }

		/* single source of truth for bwa's child rules; ADD2 is redefined per pass */
		#define DFS_GEN_CHILDREN() do { \
			if (allow_diff && xi >= indel_end_skip+tmp && len-xi >= indel_end_skip+tmp) { \
				if (e_st == STATE_M) { \
					if (e_go < max_gapo) { \
						ADD2(xi, k, l, e_mm, e_go+1, e_ge, STATE_I); \
						_Pragma("unroll") for (int j=0;j<4;++j){ uint64_t dk=c_L2[j]+SELK(j)+1, dl=c_L2[j]+SELL(j); \
							if (dk<=dl) ADD2(xi+1, dk, dl, e_mm, e_go+1, e_ge, STATE_D); } \
					} \
				} else if (e_st == STATE_I) { \
					if (e_ge < max_gape) ADD2(xi, k, l, e_mm, e_go, e_ge+1, STATE_I); \
				} else { \
					if (e_ge < max_gape && (e_ge+e_go < max_diff || occ < (uint64_t)max_del_occ)) \
						_Pragma("unroll") for (int j=0;j<4;++j){ uint64_t dk=c_L2[j]+SELK(j)+1, dl=c_L2[j]+SELL(j); \
							if (dk<=dl) ADD2(xi+1, dk, dl, e_mm, e_go, e_ge+1, STATE_D); } \
				} \
			} \
			if (allow_diff && allow_M) { \
				_Pragma("unroll") for (int j=1;j<=4;++j){ int c=(seq[xi]+j)&3; int is_mm=(j!=4||seq[xi]>3); \
					uint64_t mk=c_L2[c]+SELK(c)+1, ml=c_L2[c]+SELL(c); \
					if (mk<=ml) ADD2(xi, mk, ml, e_mm+is_mm, e_go, e_ge, STATE_M); } \
			} else if (seq[xi] < 4) { \
				int c=seq[xi]&3; uint64_t mk=c_L2[c]+SELK(c)+1, ml=c_L2[c]+SELL(c); \
				if (mk<=ml) ADD2(xi, mk, ml, e_mm, e_go, e_ge, STATE_M); \
			} \
		} while (0)

		/* pass 1: count only */
		#define ADD2(_i,_k,_l,_mm,_go,_ge,_st) do { if (STAIR_OK(_i,_mm,_go,_ge)) ++nc; } while(0)
		if (gen) DFS_GEN_CHILDREN();
		#undef ADD2

		int incl = nc;
		for (int d=1; d<32; d<<=1) { int y = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += y; }
		int total = __shfl_sync(FULL, incl, 31);
		int wp = sp + incl - nc;   /* this lane's reserved run in the shared stack */

		/* pass 2: emit directly into the shared stack */
		#define ADD2(_i,_k,_l,_mm,_go,_ge,_st) do { if (STAIR_OK(_i,_mm,_go,_ge)) { \
			sk[wp]=(_k); sl[wp]=(_l); sn[wp]=pack_node((_i),(_mm),(_go),(_ge),(_st)); ++wp; } } while(0)
		if (gen) DFS_GEN_CHILDREN();
		#undef ADD2
		#undef DFS_GEN_CHILDREN
		#undef SELK
		#undef SELL

		sp += total;
		__syncwarp();
	}
}

/* Pigeonhole prefilter: for a read of length L with max_diff errors allowed, at least one of
 * (max_diff+1) contiguous segments of length floor(L/(max_diff+1)) must exist in the reference
 * (exact FM-index match). If none do, the read is provably unmappable -> skip the DFS.
 * Bit-exact: this is a necessary condition (pigeonhole principle); zero false negatives.
 * Lanes 0..max_diff each search one segment; __any_sync aggregates. */
__device__ __forceinline__ int d_pigeonhole_prefilter(const fmidx_dev fm, const uint8_t *seq,
                                                      int len, int max_diff)
{
	const unsigned FULL = 0xffffffffu;
	int lane = threadIdx.x & 31;
	int nseg = max_diff + 1;
	int seg_len = len / nseg;
	if (seg_len < 1) return 1;
	int seg_match = 0;
	if (lane < nseg) {
		int seg_start = lane * seg_len;
		int slen = (lane == nseg - 1) ? (len - seg_start) : seg_len;
		uint64_t kk = 0, ll = fm.seq_len;
		seg_match = d_bwt_match_exact_alt_s(fm, seq + seg_start, slen, &kk, &ll);
	}
	return __any_sync(FULL, seg_match) ? 1 : 0;
}

/* persistent work-pool: warps pull reads via an atomic counter until the queue drains */
__global__ void k_dfs_warp2(fmidx_dev fm, const uint8_t *seq, const uint64_t *w_w, const int *w_bid,
                            const ReadParam *rp, int nreads, int max_gapo, int max_gape, int mode,
                            int indel_end_skip, int max_del_occ, int CAP_SM, int CAP_GL,
                            uint64_t *Gk, uint64_t *Gl, uint32_t *Gn, uint8_t *has_hit,
                            int *workctr, unsigned long long *npop, unsigned long long budget,
                            int *nflag, int wpb, uint8_t *flag_out, int use_prefilter, int *nprefilt,
                            const int *order)
{
	extern __shared__ unsigned char smem[];
	int warp_in_block = threadIdx.x >> 5, lane = threadIdx.x & 31;
	uint64_t *sk = (uint64_t*)smem + (size_t)warp_in_block * CAP_SM;
	uint64_t *sl = (uint64_t*)smem + (size_t)wpb * CAP_SM + (size_t)warp_in_block * CAP_SM;
	uint32_t *sn = (uint32_t*)((uint64_t*)smem + (size_t)wpb * CAP_SM * 2) + (size_t)warp_in_block * CAP_SM;
	int gslot = blockIdx.x * wpb + warp_in_block;
	uint64_t *gk = Gk + (size_t)gslot * CAP_GL;
	uint64_t *gl = Gl + (size_t)gslot * CAP_GL;
	uint32_t *gn = Gn + (size_t)gslot * CAP_GL;
	/* `order` (optional) is a longest-first permutation of read indices. Tree size grows
	 * exponentially with max_diff, which is a step function of read length, so handing out the
	 * expensive reads FIRST keeps the tail of the work-pool cheap (classic LPT scheduling).
	 * ncu measured SM-active/elapsed = 64%, i.e. 36% of the kernel is stragglers.
	 * Scheduling order only affects which warp takes which read -> has_hit is unchanged. */
	for (;;) {
		int idx;
		if (lane == 0) idx = atomicAdd(workctr, 1);
		idx = __shfl_sync(0xffffffffu, idx, 0);
		if (idx >= nreads) break;
		int r = order ? order[idx] : idx;
		ReadParam p = rp[r];
		if (use_prefilter && !d_pigeonhole_prefilter(fm, seq + p.seq_off, p.len, p.max_diff)) {
			if (lane == 0) { has_hit[r] = 0; npop[r] = 0; if (flag_out) flag_out[r] = 0; atomicAdd(nprefilt, 1); }
			continue;
		}
		int flagged = 0; unsigned long long nn = 0;
#ifdef DFS_INSTRUMENT
		int instr = ((r % DFS_INSTR_MOD) == 0);   /* sample reads for the (depth,errors) histogram */
#endif
		int hh = d_dfs_has_hit_warp2(fm, seq + p.seq_off, p.len, w_w + p.w_off, w_bid + p.w_off,
			p.max_diff, max_gapo, max_gape, mode, indel_end_skip, max_del_occ, sk, sl, sn, CAP_SM,
			gk, gl, gn, CAP_GL, budget, &flagged, &nn INSTR_PASS);
		if (lane == 0) { has_hit[r] = (uint8_t)hh; npop[r] = nn; if (flag_out) flag_out[r] = (uint8_t)flagged; if (flagged) atomicAdd(nflag, 1); }
	}
}

#endif /* DFS_ENGINE_CUH */
