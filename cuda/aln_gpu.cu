/* GPU bwa-aln / fused alnse (the `bwa gpualn` subcommand) -- GPU BWA-backtrack
   port for ancient DNA.
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

/* bwa-aln-gpu: full-file GPU bwa-aln (BWA-backtrack) producing a bit-exact .sai.
 *
 * Streams the FASTQ in bwa's native 0x40000-read chunks (so chunking matches the CPU driver
 * exactly). Per chunk: MT host preprocessing (bwt_cal_width + complement + per-read max_diff,
 * identical to bwa_cal_sa_reg_gap) -> warp2 GPU has_hit -> MT CPU reconcile of the ~0.015%
 * flagged/hit reads via the exact bwt_match_gap -> write the chunk's records in read order.
 * Output is byte-identical to `bwa aln -l 1024 -n 0.01 -o 2`.
 *
 * Build: make bwa-aln-gpu
 * Run:   ./bwa-aln-gpu [-l N -n F -o N -t N -f out.sai] <ref.fa> <in.fq>   (default opts as above)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" {
#include "bwt.h"
#include "bwtaln.h"
#include "bwtgap.h"
#include "bntseq.h"
#include "bwase.h"
#include "bwa.h"
#include "utils.h"
int bwt_cal_width(const bwt_t *bwt, int len, const ubyte_t *str, bwt_width_t *width);
/* in bwase.c but not in bwase.h */
void bwa_aln2seq_core(int n_aln, const bwt_aln1_t *aln, bwa_seq_t *s, int set_main, int n_multi);
void bwa_print_sam1(const bntseq_t *bns, bwa_seq_t *p, const bwa_seq_t *mate, int mode, int max_top2);
extern char *bwa_pg;   /* @PG line printed by bwa_print_sam_hdr if set */
}

#define FM_DEVICE_DEFINE_CONST
#include "fm_device.cuh"
#include "dfs_engine.cuh"
#include "scheme_engine.cuh"
#include "dual_engine.cuh"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* one chunk's host data, handed from the GPU producer thread to the CPU finisher thread */
struct Chunk {
	bwa_seq_t *seqs; int n_seqs;
	std::vector<ReadParam> rp;
	std::vector<uint8_t> seq_flat; std::vector<uint64_t> w_flat; std::vector<int> bid_flat;
	std::vector<uint8_t> has_hit;
	std::vector<int> order;                 /* longest-first work-pool permutation */
	std::vector<unsigned long long> npop;   /* GPUALN_HISTO only */
	std::vector<uint8_t> flag;              /* GPUALN_HISTO only */
	gap_opt_t base; int stack_maxdiff, max_len;
	int seq_id;
};

/* per-GPU context: own BWT copy, backing, device buffers, occupancy */
struct GpuCtx {
	int dev;
	fmidx_dev fm;
	int nblocks, bdim, wpb, CAP_SM, CAP_GL;
	size_t shbytes;
	uint64_t *Gk, *Gl; uint32_t *Gn; uint32_t *Gdual;
	uint8_t *d_seq; uint64_t *d_ww; int *d_wbid; ReadParam *d_rp;
	uint8_t *d_hit; unsigned long long *d_npop; int *d_wc, *d_nflag, *d_nprefilt;
	uint8_t *d_flag; int *d_order;
	size_t cap_seq, cap_w, cap_n;
};

/* C-callable entry: usable as a bwa subcommand (`bwa gpualn ...`) or standalone (ALN_GPU_MAIN).
 * argv[0] is the program/subcommand name; options are parsed from argv[1..] (bwa convention). */
extern "C" int bwa_alnse_gpu(int argc, char **argv)
{
	gap_opt_t *opt = gap_init_opt();
	opt->seed_len = 1024; opt->fnr = 0.01; opt->max_diff = -1; opt->max_gapo = 2; opt->n_threads = 16;
	const char *out_fn = NULL; int wpb = 4, CAP_SM = 512, CAP_GL = 16384;
	int sam_mode = 0, n_occ = 3; char *rg_line = NULL;   /* -S: fused alnse (SAM out); -r: RG */
	int c;
	while ((c = getopt(argc, argv, "l:n:o:t:f:Sr:")) >= 0) {
		if (c=='l') opt->seed_len = atoi(optarg);
		else if (c=='n') { if (strstr(optarg,".")) { opt->fnr=atof(optarg); opt->max_diff=-1; } else { opt->max_diff=atoi(optarg); opt->fnr=-1; } }
		else if (c=='o') opt->max_gapo = atoi(optarg);
		else if (c=='t') opt->n_threads = atoi(optarg);
		else if (c=='f') out_fn = optarg;
		else if (c=='S') sam_mode = 1;
		else if (c=='r') { if ((rg_line = bwa_set_rg(optarg)) == 0) return 1; }
	}
	if (optind + 2 > argc) { fprintf(stderr, "usage: %s [-l N -n F -o N -t N -f out.sai] <ref.fa> <in.fq>\n", argv[0]); return 1; }
	const char *prefix = argv[optind], *fq = argv[optind+1];
	if (getenv("DFS_WARP_CAP")) CAP_SM = atoi(getenv("DFS_WARP_CAP"));
	if (getenv("DFS_WARP_GCAP")) CAP_GL = atoi(getenv("DFS_WARP_GCAP"));
	if (getenv("DFS_WARP_WPB")) wpb = atoi(getenv("DFS_WARP_WPB"));   /* warps per block (occupancy sweep) */
	/* The two-level stack spills in CHUNK=128 blocks and assumes a full chunk plus a wave's worth
	 * of headroom fits in the shared window; below that the spill silently corrupts the frontier
	 * (CAP_SM=128 produced a WRONG .sai). Enforce it rather than trusting the caller. */
	if (CAP_SM < 256) { fprintf(stderr, "[aln-gpu] CAP_SM=%d too small (must be >= 256)\n", CAP_SM); return 1; }
	if (CAP_GL < 256 || (CAP_GL % 128)) { fprintf(stderr, "[aln-gpu] CAP_GL=%d invalid (>=256, multiple of 128)\n", CAP_GL); return 1; }
	/* the dual engine packs k and l into 33 bits each; that holds only up to a ~4.29 Gbp reference */
	int dual_guard_pending = 1;
	unsigned long long budget = getenv("DFS_BUDGET") ? strtoull(getenv("DFS_BUDGET"),NULL,10) : 2000000ULL;
	int use_prefilter = getenv("DFS_NOPREFILTER") ? 0 : 1;
	int nT = opt->n_threads > 0 ? opt->n_threads : 1;
	int do_histo = getenv("GPUALN_HISTO") != NULL;
	int use_order = getenv("GPUALN_NOORDER") ? 0 : 1;
	int use_scheme = getenv("GPUALN_SCHEME") ? 1 : 0;
	int use_dual = getenv("GPUALN_DUAL") ? 1 : 0;   /* two reads/warp -> per-lane MLP=2 */   /* bidirectional search-scheme engine */   /* longest-first scheduling (A/B knob) */   /* opt-in per-length-band node-pop/flag histogram */

	char bwt_fn[4096]; snprintf(bwt_fn, sizeof bwt_fn, "%s.bwt", prefix);
	fprintf(stderr, "[aln-gpu] loading %s\n", bwt_fn);
	bwt_t *bwt = bwt_restore_bwt(bwt_fn);
	if (!bwt) { fprintf(stderr, "failed to load bwt\n"); return 1; }
	if (use_dual && !n12_fits(bwt->seq_len)) {
		fprintf(stderr, "[aln-gpu] GPUALN_DUAL needs seq_len < 2^33 (have %llu); falling back to the 20-byte engine\n",
		        (unsigned long long)bwt->seq_len);
		use_dual = 0;
	}
	(void)dual_guard_pending;

	/* multi-GPU init: detect devices, upload BWT + allocate backing on each */
	int nGpu = 0; CK(cudaGetDeviceCount(&nGpu));
	if (getenv("GPUALN_NGPU")) nGpu = atoi(getenv("GPUALN_NGPU"));
	if (nGpu < 1) nGpu = 1;
	std::vector<GpuCtx> gpus(nGpu);
	for (int g = 0; g < nGpu; g++) {
		CK(cudaSetDevice(g));
		GpuCtx &gx = gpus[g];
		gx.dev = g; gx.wpb = wpb; gx.CAP_SM = CAP_SM; gx.CAP_GL = CAP_GL;
		uint32_t *db = NULL;
		CK(cudaMalloc(&db, bwt->bwt_size * sizeof(uint32_t)));
		CK(cudaMemcpy(db, bwt->bwt, bwt->bwt_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
		CK(cudaMemcpyToSymbol(c_cnt_table, bwt->cnt_table, sizeof(uint32_t)*256));
		CK(cudaMemcpyToSymbol(c_L2, bwt->L2, sizeof(uint64_t)*5));
		gx.fm = fmidx_dev{ db, bwt->primary, bwt->seq_len };
		int numSM = 0; CK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, g));
		gx.bdim = wpb * 32;
		/* dual: two 12-byte-node stacks per warp; that is what buys 2 reads/warp at 8 warps/SM */
		gx.shbytes = use_dual ? (size_t)wpb * 2 * CAP_SM * 12
		                      : (size_t)wpb * CAP_SM * (use_scheme ? sizeof(SNode) : 20);
		int mb = 0;
		if (use_dual) {
			CK(cudaFuncSetAttribute(k_dfs_warp2_dual, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gx.shbytes));
			CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_dfs_warp2_dual, gx.bdim, gx.shbytes));
		} else if (use_scheme) {
			if (sch_upload(g == 0)) { fprintf(stderr, "[scheme] table validation FAILED\n"); return 1; }
			CK(cudaFuncSetAttribute(k_dfs_scheme, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gx.shbytes));
			CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_dfs_scheme, gx.bdim, gx.shbytes));
		} else {
			CK(cudaFuncSetAttribute(k_dfs_warp2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gx.shbytes));
			CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_dfs_warp2, gx.bdim, gx.shbytes));
		}
		gx.nblocks = mb > 0 ? mb * numSM : numSM;
		size_t nwarps = (size_t)gx.nblocks * wpb;
		gx.Gdual = NULL;
		if (use_dual) CK(cudaMalloc(&gx.Gdual, nwarps*2*(size_t)CAP_GL*3*4));
		else { CK(cudaMalloc(&gx.Gk, nwarps*CAP_GL*8)); CK(cudaMalloc(&gx.Gl, nwarps*CAP_GL*8)); CK(cudaMalloc(&gx.Gn, nwarps*CAP_GL*4)); }
		gx.d_seq=NULL; gx.d_ww=NULL; gx.d_wbid=NULL; gx.d_rp=NULL;
		gx.d_hit=NULL; gx.d_npop=NULL; gx.d_wc=NULL; gx.d_nflag=NULL; gx.d_nprefilt=NULL; gx.d_flag=NULL; gx.d_order=NULL;
		gx.cap_seq=0; gx.cap_w=0; gx.cap_n=0;
		char name[256]; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, g));
		snprintf(name, sizeof name, "%s", prop.name);
		fprintf(stderr, "[aln-gpu] GPU %d (%s): %d SM, %d blk/SM x %d warps, backing %.0f MB\n",
			g, name, numSM, mb, mb*wpb, nwarps*CAP_GL*(use_dual?24.0:20.0)/1e6);
	}
	fprintf(stderr, "[aln-gpu] %d GPU(s), CAP_SM=%d CAP_GL=%d; %d CPU threads\n", nGpu, CAP_SM, CAP_GL, nT);

	/* output: .sai (default) or fused alnse SAM (-S) */
	FILE *out = NULL; bntseq_t *bns = NULL; ubyte_t *pacseq = NULL;
	if (sam_mode) {
		bwase_initialize();
		bns = bns_restore(prefix);
		srand48(bns->seed);                       /* exact samse RNG seeding (repeat-hit selection) */
		char sa_fn[4096]; snprintf(sa_fn, sizeof sa_fn, "%s.sa", prefix);
		bwt_restore_sa(sa_fn, bwt);                /* SA for bwa_sa2pos; bwt kept (not destroyed) */
		pacseq = (ubyte_t*)calloc(bns->l_pac/4+1, 1);
		err_rewind(bns->fp_pac); err_fread_noeof(pacseq, 1, bns->l_pac/4+1, bns->fp_pac);
		if (!bwa_pg) { /* @PG provenance (standalone only; as a bwa subcommand, main.c sets bwa_pg) */
			char pg[8192]; int o = snprintf(pg, sizeof pg, "@PG\tID:bwa-aln-gpu\tPN:bwa-aln-gpu\tVN:gpu\tCL:");
			for (int i=0;i<argc && o<(int)sizeof pg-2;i++) o += snprintf(pg+o, sizeof pg-o, "%s%s", i?" ":"", argv[i]);
			bwa_pg = strdup(pg);
		}
		bwa_print_sam_hdr(bns, rg_line);           /* @HD/@SQ/@RG/@PG header to stdout */
		fprintf(stderr, "[aln-gpu] fused alnse (SAM) mode; SA+pac loaded\n");
	} else {
		out = out_fn ? fopen(out_fn, "wb") : stdout;
		if (!out) { fprintf(stderr, "cannot open %s\n", out_fn); return 1; }
		err_fwrite(SAI_MAGIC, 1, 4, out);
		err_fwrite(opt, sizeof(gap_opt_t), 1, out);
	}

	bwa_seqio_t *ks = bwa_seq_open(fq);

	long long tot=0, tot_flag=0, tot_prefilt=0; double t0=now_s();

	/* opt-in instrumentation (GPUALN_HISTO=1) */
	const int MAXL = 256;
	std::vector<unsigned long long> H_n(MAXL,0), H_hit(MAXL,0), H_flag(MAXL,0), H_sum(MAXL,0), H_max(MAXL,0);
	std::vector<int> H_d(MAXL,-1);
	double gpu_work_s = 0, recon_s = 0;

	/* ---- Multi-GPU pipeline (3 stages):
	 * Stage 1 (main thread): read FASTQ + MT preprocess -> push Chunk* to "ready" queue.
	 * Stage 2 (N GPU workers): pop from ready -> upload -> kernel -> download has_hit -> mark done.
	 * Stage 3 (finisher): process completed chunks IN ORDER -> reconcile + output.
	 * Ordering: chunks have seq_id; finisher waits for next-in-order via completion buffer.
	 * Bit-exactness: single ordered finisher preserves drand48/output order. ---- */
	std::mutex rmu; std::condition_variable r_ready, r_free;
	std::queue<Chunk*> RQ; bool read_done = false;
	const size_t RQCAP = (size_t)(nGpu + 2);

	std::mutex cmu; std::condition_variable c_done_cv;
	std::vector<Chunk*> cslots; int next_out = 0; bool all_done = false;

	auto gpu_worker = [&](int gid){
		GpuCtx &gx = gpus[gid];
		CK(cudaSetDevice(gx.dev));
		for (;;) {
			Chunk *c;
			{ std::unique_lock<std::mutex> lk(rmu);
			  r_ready.wait(lk, [&]{ return !RQ.empty() || read_done; });
			  if (RQ.empty()) break;
			  c = RQ.front(); RQ.pop(); }
			r_free.notify_one();
			int nseq = c->n_seqs;
			size_t so = c->seq_flat.size(), wo = c->w_flat.size();
			if (so > gx.cap_seq){ if(gx.d_seq)cudaFree(gx.d_seq); CK(cudaMalloc(&gx.d_seq, so)); gx.cap_seq=so; }
			if (wo > gx.cap_w){ if(gx.d_ww)cudaFree(gx.d_ww); if(gx.d_wbid)cudaFree(gx.d_wbid);
				CK(cudaMalloc(&gx.d_ww, wo*8)); CK(cudaMalloc(&gx.d_wbid, wo*4)); gx.cap_w=wo; }
			if ((size_t)nseq > gx.cap_n){ if(gx.d_rp)cudaFree(gx.d_rp); if(gx.d_hit)cudaFree(gx.d_hit);
				if(gx.d_npop)cudaFree(gx.d_npop); if(gx.d_flag)cudaFree(gx.d_flag);
				CK(cudaMalloc(&gx.d_rp, nseq*sizeof(ReadParam))); CK(cudaMalloc(&gx.d_hit, nseq));
				CK(cudaMalloc(&gx.d_npop, nseq*8)); CK(cudaMalloc(&gx.d_flag, nseq));
				if (gx.d_order) cudaFree(gx.d_order); CK(cudaMalloc(&gx.d_order, nseq*4)); gx.cap_n=nseq; }
			if (!gx.d_wc){ CK(cudaMalloc(&gx.d_wc,4)); CK(cudaMalloc(&gx.d_nflag,4)); CK(cudaMalloc(&gx.d_nprefilt,4)); }
			CK(cudaMemcpy(gx.d_seq, c->seq_flat.data(), so, cudaMemcpyHostToDevice));
			CK(cudaMemcpy(gx.d_ww, c->w_flat.data(), wo*8, cudaMemcpyHostToDevice));
			CK(cudaMemcpy(gx.d_wbid, c->bid_flat.data(), wo*4, cudaMemcpyHostToDevice));
			CK(cudaMemcpy(gx.d_rp, c->rp.data(), nseq*sizeof(ReadParam), cudaMemcpyHostToDevice));
			CK(cudaMemcpy(gx.d_order, c->order.data(), nseq*4, cudaMemcpyHostToDevice));
			CK(cudaMemset(gx.d_wc,0,4)); CK(cudaMemset(gx.d_nflag,0,4)); CK(cudaMemset(gx.d_nprefilt,0,4));
			double _gk0 = now_s();
			if (use_dual)
				k_dfs_warp2_dual<<<gx.nblocks, gx.bdim, gx.shbytes>>>(gx.fm, gx.d_seq, gx.d_ww, gx.d_wbid, gx.d_rp, nseq,
					c->base.max_gapo, c->base.max_gape, c->base.mode, c->base.indel_end_skip, c->base.max_del_occ,
					gx.CAP_SM, gx.CAP_GL, gx.Gdual, gx.d_hit, gx.d_wc, gx.d_npop, budget, gx.d_nflag, gx.wpb,
					gx.d_flag, use_order ? gx.d_order : NULL);
			else if (use_scheme)
				k_dfs_scheme<<<gx.nblocks, gx.bdim, gx.shbytes>>>(gx.fm, gx.d_seq, gx.d_ww, gx.d_wbid, gx.d_rp, nseq,
					c->base.max_gapo, c->base.max_gape, c->base.mode, c->base.indel_end_skip, c->base.max_del_occ,
					gx.CAP_SM, gx.CAP_GL, gx.Gk, gx.Gl, gx.Gn, gx.d_hit, gx.d_wc, gx.d_npop, budget, gx.d_nflag,
					gx.wpb, gx.d_flag, use_order ? gx.d_order : NULL, gx.d_nprefilt);
			else
				k_dfs_warp2<<<gx.nblocks, gx.bdim, gx.shbytes>>>(gx.fm, gx.d_seq, gx.d_ww, gx.d_wbid, gx.d_rp, nseq,
					c->base.max_gapo, c->base.max_gape, c->base.mode, c->base.indel_end_skip, c->base.max_del_occ,
					gx.CAP_SM, gx.CAP_GL, gx.Gk, gx.Gl, gx.Gn, gx.d_hit, gx.d_wc, gx.d_npop, budget, gx.d_nflag,
					gx.wpb, gx.d_flag, use_prefilter, gx.d_nprefilt, use_order ? gx.d_order : NULL);
			CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
			if (do_histo) { std::lock_guard<std::mutex> lk(cmu); gpu_work_s += now_s() - _gk0; }
			int hpf=0; CK(cudaMemcpy(&hpf, gx.d_nprefilt, 4, cudaMemcpyDeviceToHost));
			CK(cudaMemcpy(c->has_hit.data(), gx.d_hit, nseq, cudaMemcpyDeviceToHost));
			if (do_histo) {   /* node-pop / flag detail for the per-length-band report */
				c->npop.resize(nseq); c->flag.resize(nseq);
				CK(cudaMemcpy(c->npop.data(), gx.d_npop, (size_t)nseq*8, cudaMemcpyDeviceToHost));
				CK(cudaMemcpy(c->flag.data(), gx.d_flag, nseq, cudaMemcpyDeviceToHost));
			}
			{ std::lock_guard<std::mutex> lk(cmu);
			  tot_prefilt += hpf;
			  cslots[c->seq_id] = c;
			  c_done_cv.notify_one(); }
		}
	};

	auto finisher = [&](){
		for (;;) {
			Chunk *c = NULL;
			{ std::unique_lock<std::mutex> lk(cmu);
			  c_done_cv.wait(lk, [&]{ return (next_out < (int)cslots.size() && cslots[next_out]) || all_done; });
			  if (next_out >= (int)cslots.size() && all_done) break;
			  if (!cslots[next_out]) continue;
			  c = cslots[next_out]; cslots[next_out] = NULL; next_out++; }
			int nseq = c->n_seqs;
			std::vector<int> idx; for (int i=0;i<nseq;i++) if (c->has_hit[i]) idx.push_back(i);
			tot_flag += idx.size();
			if (do_histo && !c->npop.empty()) {   /* accumulate the per-length-band routing histogram */
				for (int i=0;i<nseq;i++) {
					int L = c->rp[i].len; if (L < 0 || L >= MAXL) continue;
					H_n[L]++; H_d[L] = c->rp[i].max_diff;
					if (c->has_hit[i]) H_hit[L]++;
					if (c->flag[i]) H_flag[L]++;
					H_sum[L] += c->npop[i];
					if (c->npop[i] > H_max[L]) H_max[L] = c->npop[i];
				}
			}
			std::vector<int> n_aln(nseq,0); std::vector<bwt_aln1_t*> aln(nseq,NULL);
			std::vector<std::thread> ths;
			double _rc0 = now_s();
			for (int t=0;t<nT;t++) ths.emplace_back([&,t](){
				gap_stack_t *st = gap_init_stack(c->stack_maxdiff, c->base.max_gapo, c->base.max_gape, &c->base);
				std::vector<bwt_width_t> w(c->max_len+1);
				for (size_t x=t;x<idx.size();x+=nT){
					int i=idx[x], len=c->rp[i].len;
					for (int j=0;j<=len;j++){ w[j].w=c->w_flat[c->rp[i].w_off+j]; w[j].bid=c->bid_flat[c->rp[i].w_off+j]; }
					gap_opt_t lo=c->base; lo.max_diff=c->rp[i].max_diff; lo.seed_len = opt->seed_len<len?opt->seed_len:0x7fffffff;
					int na=0; aln[i]=bwt_match_gap(bwt, len, c->seq_flat.data()+c->rp[i].seq_off, w.data(), (bwt_width_t*)0, &lo, &na, st);
					n_aln[i]=na;
				}
				gap_destroy_stack(st);
			});
			for (auto&th:ths) th.join();
			if (do_histo) recon_s += now_s() - _rc0;
			if (!sam_mode) {
				for (int i=0;i<nseq;i++){ err_fwrite(&n_aln[i],4,1,out); if (n_aln[i]) err_fwrite(aln[i],sizeof(bwt_aln1_t),n_aln[i],out); free(aln[i]); }
			} else {
				for (int i=0;i<nseq;i++){ bwa_aln2seq_core(n_aln[i], aln[i], &c->seqs[i], 1, n_occ); free(aln[i]); }
				ths.clear();
				for (int t=0;t<nT;t++) ths.emplace_back([&,t](){
					for (int i=t;i<nseq;i+=nT){ bwa_seq_t *p=&c->seqs[i];
						bwa_cal_pac_pos_core(bns, bwt, p, opt->max_diff, opt->fnr);
						int strand, nm2=0;
						for (int j=0;j<p->n_multi;j++){ bwt_multi1_t *q=p->multi+j;
							q->pos = bwa_sa2pos(bns, bwt, q->pos, p->len+q->ref_shift, &strand); q->strand=strand;
							if (q->pos != p->pos && q->pos != (bwtint_t)-1) p->multi[nm2++]=*q; }
						p->n_multi=nm2; }
				});
				for (auto&th:ths) th.join();
				bwa_refine_gapped(bns, nseq, c->seqs, pacseq);
				for (int i=0;i<nseq;i++) bwa_print_sam1(bns, &c->seqs[i], 0, opt->mode, opt->max_top2);
			}
			tot += nseq;
			bwa_free_read_seq(nseq, c->seqs);
			delete c;
			fprintf(stderr, "\r[aln-gpu] %lld reads done (%.0f reads/s)   ", tot, tot/(now_s()-t0));
		}
	};

	std::vector<std::thread> workers;
#ifdef DFS_INSTRUMENT
	/* Idea-A cost probe: DFS_STAIR=p:u1:u2 emulates ONE search of a p-part staircase scheme.
	 * Output is MEANINGLESS with this set -- it measures that search's tree size only. */
	if (getenv("DFS_STAIR")) {
		int sp_=3, su1=1, su2=2, on=1;
		sscanf(getenv("DFS_STAIR"), "%d:%d:%d", &sp_, &su1, &su2);
		for (int g=0; g<nGpu; g++) {
			CK(cudaSetDevice(gpus[g].dev));
			CK(cudaMemcpyToSymbol(g_stair_on, &on, 4));
			CK(cudaMemcpyToSymbol(g_stair_p, &sp_, 4));
			CK(cudaMemcpyToSymbol(g_stair_u1, &su1, 4));
			CK(cudaMemcpyToSymbol(g_stair_u2, &su2, 4));
		}
		fprintf(stderr, "[stair] p=%d U=(%d,%d,max_diff)  *** .sai is INVALID; cost measurement only ***\n", sp_, su1, su2);
	}
#endif
	for (int g=0;g<nGpu;g++) workers.emplace_back(gpu_worker, g);
	std::thread cons(finisher);

	int n_seqs; bwa_seq_t *seqs; int seq_id = 0;
	while ((seqs = bwa_read_seq(ks, 0x40000, &n_seqs, opt->mode, opt->trim_qual)) != 0) {
		Chunk *c = new Chunk();
		c->seqs = seqs; c->n_seqs = n_seqs; c->base = *opt; c->max_len = 0; c->seq_id = seq_id++;
		for (int i=0;i<n_seqs;i++) if (seqs[i].len > c->max_len) c->max_len = seqs[i].len;
		if (opt->fnr > 0.0) c->base.max_diff = bwa_cal_maxdiff(c->max_len, BWA_AVG_ERR, opt->fnr);
		if (c->base.max_diff < c->base.max_gapo) c->base.max_gapo = c->base.max_diff;
		c->stack_maxdiff = c->base.max_diff;
		c->rp.resize(n_seqs);
		size_t so=0, wo=0;
		for (int i=0;i<n_seqs;i++){ c->rp[i].seq_off=so; c->rp[i].w_off=wo; c->rp[i].len=seqs[i].len; so+=seqs[i].len; wo+=seqs[i].len+1; }
		c->seq_flat.resize(so); c->w_flat.resize(wo); c->bid_flat.resize(wo); c->has_hit.resize(n_seqs);
		/* longest-first: max_diff (hence tree size) is a step function of read length */
		c->order.resize(n_seqs);
		for (int i=0;i<n_seqs;i++) c->order[i]=i;
		std::sort(c->order.begin(), c->order.end(),
		          [&](int a, int b){ return seqs[a].len > seqs[b].len; });

		std::vector<std::thread> ths;
		for (int t=0;t<nT;t++) ths.emplace_back([&,t](){
			std::vector<bwt_width_t> w(c->max_len+1);
			for (int i=t;i<n_seqs;i+=nT){ bwa_seq_t *p=&seqs[i];
				memset(w.data(), 0, (p->len+1)*sizeof(bwt_width_t));
				bwt_cal_width(bwt, p->len, p->seq, w.data());
				c->rp[i].max_diff = (opt->fnr>0.0)? bwa_cal_maxdiff(p->len, BWA_AVG_ERR, opt->fnr) : c->base.max_diff;
				for (int j=0;j<p->len;j++) c->seq_flat[c->rp[i].seq_off+j] = p->seq[j]>3?4:3-p->seq[j];
				for (int j=0;j<=p->len;j++){ c->w_flat[c->rp[i].w_off+j]=w[j].w; c->bid_flat[c->rp[i].w_off+j]=w[j].bid; }
			}
		});
		for (auto&th:ths) th.join();

		{ std::unique_lock<std::mutex> lk(cmu); if (seq_id > (int)cslots.size()) cslots.resize(seq_id, NULL); }
		{ std::unique_lock<std::mutex> lk(rmu); r_free.wait(lk, [&]{ return RQ.size() < RQCAP; }); RQ.push(c); }
		r_ready.notify_one();
	}
	{ std::lock_guard<std::mutex> lk(rmu); read_done = true; } r_ready.notify_all();
	for (auto&w:workers) w.join();
	{ std::lock_guard<std::mutex> lk(cmu); all_done = true; } c_done_cv.notify_one();
	cons.join();
	double total = now_s()-t0;
	fprintf(stderr, "\n[aln-gpu] DONE: %lld reads in %.1f s = %.0f reads/s; flagged->CPU %lld (%.3f%%); %s %lld (%.2f%%)\n",
		tot, total, tot/total, tot_flag, 100.0*tot_flag/tot,
		use_scheme ? "scheme-fallback" : "prefiltered", tot_prefilt, tot>0?100.0*tot_prefilt/tot:0.0);

	if (do_histo) {   /* per-length-band routing report (budget = %llu) */
		fprintf(stderr, "[histo] budget=%llu  GPU-kernel %.3fs  CPU-reconcile %.3fs  (reconcile/GPU = %.2fx; <1 means it hides under overlap)\n",
			budget, gpu_work_s, recon_s, gpu_work_s>0 ? recon_s/gpu_work_s : 0.0);
		fprintf(stderr, "[histo] %4s %3s %12s %7s %8s %12s %12s\n", "len","d","reads","hit%","flag%","mean_pops","max_pops");
		unsigned long long agg_n=0, agg_flag=0;
		for (int L=0; L<MAXL; L++) {
			if (!H_n[L]) continue;
			agg_n += H_n[L]; agg_flag += H_flag[L];
			fprintf(stderr, "[histo] %4d %3d %12llu %7.2f %8.4f %12.0f %12llu\n",
				L, H_d[L], H_n[L], 100.0*H_hit[L]/H_n[L], 100.0*H_flag[L]/H_n[L],
				(double)H_sum[L]/H_n[L], H_max[L]);
		}
		fprintf(stderr, "[histo] TOTAL reads=%llu  overall flag%%=%.4f\n", agg_n, agg_n? 100.0*agg_flag/agg_n : 0.0);
		unsigned long long agg_pops=0; for (int L=0;L<MAXL;L++) agg_pops += H_sum[L];
		fprintf(stderr, "[histo] TOTAL node-pops=%llu  (%.0f/read)  = %.2f G-pops/s over the %.1fs kernel\n",
			agg_pops, agg_n? (double)agg_pops/agg_n : 0.0, gpu_work_s>0? agg_pops/gpu_work_s/1e9 : 0.0, gpu_work_s);
#ifdef DFS_INSTRUMENT
		{	/* FM-probe accounting + (depth, errors) profile of node pops */
			unsigned long long pr=0, bu=0, po=0, wv=0, spl=0;
			CK(cudaMemcpyFromSymbol(&wv, g_waves, 8));
			CK(cudaMemcpyFromSymbol(&spl, g_spills, 8));
			static unsigned long long dh[DHIST_D*DHIST_E];
			CK(cudaMemcpyFromSymbol(&pr, g_probes, 8));
			CK(cudaMemcpyFromSymbol(&bu, g_buckets, 8));
			CK(cudaMemcpyFromSymbol(&po, g_pops, 8));
			CK(cudaMemcpyFromSymbol(dh, g_dhist, sizeof(dh)));
			fprintf(stderr, "[instr] pops=%llu  probes=%llu (%.3f/pop)  buckets=%llu (%.3f/probe)\n",
				po, pr, po? (double)pr/po : 0.0, bu, pr? (double)bu/pr : 0.0);
			fprintf(stderr, "[instr] waves=%llu  MEAN ACTIVE LANES = %.2f / 32 (%.0f%% lane utilisation)  spills=%llu (%.3f/wave)\n",
				wv, wv? (double)po/wv : 0.0, wv? 100.0*((double)po/wv)/32.0 : 0.0, spl, wv? (double)spl/wv : 0.0);
			fprintf(stderr, "[instr] effective %.3f G-occ4/s vs 2.32 G/s random-gather ceiling = %.0f%% of ceiling\n",
				gpu_work_s>0? bu/gpu_work_s/1e9 : 0.0, gpu_work_s>0? 100.0*(bu/gpu_work_s/1e9)/2.32 : 0.0);
			unsigned long long tot_h=0; for (int i=0;i<DHIST_D*DHIST_E;i++) tot_h += dh[i];
			fprintf(stderr, "[instr] node pops by (depth = read bases consumed, errors used); sampled 1/%d reads, n=%llu\n",
				DFS_INSTR_MOD, tot_h);
			fprintf(stderr, "[instr] %5s %10s %6s %6s", "depth", "pops", "%", "cum%");
			for (int e=0;e<DHIST_E;e++) fprintf(stderr, " %7s%d", "e=", e);
			fprintf(stderr, "\n");
			double cum = 0;
			for (int d=0; d<DHIST_D; d++) {
				unsigned long long row=0; for (int e=0;e<DHIST_E;e++) row += dh[d*DHIST_E+e];
				if (!row) continue;
				double pct = tot_h? 100.0*row/tot_h : 0.0; cum += pct;
				fprintf(stderr, "[instr] %5d %10llu %6.2f %6.2f", d, row, pct, cum);
				for (int e=0;e<DHIST_E;e++) fprintf(stderr, " %8llu", dh[d*DHIST_E+e]);
				fprintf(stderr, "\n");
			}
		}
#endif
	}

	if (out_fn) fclose(out);
	bwa_seq_close(ks);
	bwt_destroy(bwt);
	return 0;
}

#ifdef ALN_GPU_MAIN
int main(int argc, char **argv) { return bwa_alnse_gpu(argc, argv); }
#endif
