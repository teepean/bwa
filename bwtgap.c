#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bwtgap.h"
#define BWT_GAP_BATCH_MAX 8
#include "bwtaln.h"

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif

#define STATE_M 0
#define STATE_I 1
#define STATE_D 2

#ifdef ALN_PROFILE
/* Phase-0 instrumentation: per-read search-tree size distribution.
 * Guarded so the normal build and the eventual CUDA port stay clean.
 * Histograms use log2 bins; updated with GCC atomics (thread-safe under -t N). */
#include <stdatomic.h>
#define ALN_PROF_BINS 40
static atomic_llong g_prof_peak_hist[ALN_PROF_BINS];   // bins of peak stack->n_entries
static atomic_llong g_prof_pop_hist[ALN_PROF_BINS];    // bins of nodes expanded (pops)
static atomic_llong g_prof_n_reads, g_prof_n_cap, g_prof_n_zero;
static atomic_llong g_prof_sum_peak, g_prof_sum_pop, g_prof_max_peak, g_prof_max_pop;

static inline int aln_prof_bin(long long v){ int b=0; while(v){ ++b; v>>=1; } return b<ALN_PROF_BINS?b:ALN_PROF_BINS-1; }

static void aln_prof_record(long long peak, long long pop, int hit_cap, int n_aln)
{
	atomic_fetch_add(&g_prof_n_reads, 1);
	atomic_fetch_add(&g_prof_peak_hist[aln_prof_bin(peak)], 1);
	atomic_fetch_add(&g_prof_pop_hist[aln_prof_bin(pop)], 1);
	atomic_fetch_add(&g_prof_sum_peak, peak);
	atomic_fetch_add(&g_prof_sum_pop, pop);
	if (hit_cap) atomic_fetch_add(&g_prof_n_cap, 1);
	if (n_aln == 0) atomic_fetch_add(&g_prof_n_zero, 1);
	long long m;
	m = atomic_load(&g_prof_max_peak); while (peak > m && !atomic_compare_exchange_weak(&g_prof_max_peak,&m,peak));
	m = atomic_load(&g_prof_max_pop);  while (pop  > m && !atomic_compare_exchange_weak(&g_prof_max_pop,&m,pop));
}

__attribute__((destructor)) static void aln_prof_dump(void)
{
	long long n = atomic_load(&g_prof_n_reads);
	if (!n) return;
	fprintf(stderr, "\n=== ALN_PROFILE: search-tree size over %lld reads ===\n", n);
	fprintf(stderr, "reads with 0 hits: %lld (%.1f%%)   reads hitting max_entries cap: %lld (%.2f%%)\n",
			atomic_load(&g_prof_n_zero), 100.0*atomic_load(&g_prof_n_zero)/n,
			atomic_load(&g_prof_n_cap), 100.0*atomic_load(&g_prof_n_cap)/n);
	fprintf(stderr, "peak queue entries: mean %.1f  max %lld\n", (double)atomic_load(&g_prof_sum_peak)/n, atomic_load(&g_prof_max_peak));
	fprintf(stderr, "nodes expanded/read: mean %.1f  max %lld\n", (double)atomic_load(&g_prof_sum_pop)/n, atomic_load(&g_prof_max_pop));
	fprintf(stderr, "peak-entries histogram (log2 bin = #entries in [2^(b-1),2^b) ):\n");
	for (int b=0;b<ALN_PROF_BINS;b++){ long long c=atomic_load(&g_prof_peak_hist[b]); if(c) fprintf(stderr,"  <=2^%-2d (%8lld): %lld\n", b, (b?1LL<<b:0), c); }
	fprintf(stderr, "nodes-expanded histogram:\n");
	for (int b=0;b<ALN_PROF_BINS;b++){ long long c=atomic_load(&g_prof_pop_hist[b]); if(c) fprintf(stderr,"  <=2^%-2d (%8lld): %lld\n", b, (b?1LL<<b:0), c); }
}
#endif

#define aln_score(m,o,e,p) ((m)*(p)->s_mm + (o)*(p)->s_gapo + (e)*(p)->s_gape)

gap_stack_t *gap_init_stack2(int max_score)
{
	gap_stack_t *stack;
	stack = (gap_stack_t*)calloc(1, sizeof(gap_stack_t));
	stack->n_stacks = max_score;
	stack->stacks = (gap_stack1_t*)calloc(stack->n_stacks, sizeof(gap_stack1_t));
	return stack;
}

gap_stack_t *gap_init_stack(int max_mm, int max_gapo, int max_gape, const gap_opt_t *opt)
{
	return gap_init_stack2(aln_score(max_mm+1, max_gapo+1, max_gape+1, opt));
}

void gap_destroy_stack(gap_stack_t *stack)
{
	int i;
	for (i = 0; i != stack->n_stacks; ++i) free(stack->stacks[i].stack);
	free(stack->stacks);
	free(stack);
}

static void gap_reset_stack(gap_stack_t *stack)
{
	int i;
	for (i = 0; i != stack->n_stacks; ++i)
		stack->stacks[i].n_entries = 0;
	stack->best = stack->n_stacks;
	stack->n_entries = 0;
}

/* PREFETCH: a child's next bwt_2occ4 will touch the Occ buckets holding k-1 and l, and those
 * addresses are already known here. Within a score bin the queue is LIFO, so a child pushed at
 * the CURRENT best score is popped immediately next -- that is the match-continuation path, the
 * most common child -- making this a short, well-targeted prefetch distance. The search is
 * latency-bound on random probes into a 3.14 GB index, so this is the cheapest lever available.
 * Prefetch never faults, but skip k==0 so we do not touch a wild address for (bwtint_t)-1. */
static inline void gap_push(const bwt_t *bwt, gap_stack_t *stack, int i, bwtint_t k, bwtint_t l, int n_mm, int n_gapo, int n_gape, int n_ins, int n_del,
							int state, int is_diff, const gap_opt_t *opt)
{
	int score;
	if (k) __builtin_prefetch(bwt_occ_intv(bwt, k - 1), 0, 3);
	__builtin_prefetch(bwt_occ_intv(bwt, l), 0, 3);
	gap_entry_t *p;
	gap_stack1_t *q;
	score = aln_score(n_mm, n_gapo, n_gape, opt);
	q = stack->stacks + score;
	if (q->n_entries == q->m_entries) {
		q->m_entries = q->m_entries? q->m_entries<<1 : 4;
		q->stack = (gap_entry_t*)realloc(q->stack, sizeof(gap_entry_t) * q->m_entries);
	}
	p = q->stack + q->n_entries;
	p->info = (uint32_t)score<<21 | i; p->k = k; p->l = l;
	p->n_mm = n_mm; p->n_gapo = n_gapo; p->n_gape = n_gape;
	p->n_ins = n_ins; p->n_del = n_del;
	p->state = state; 
	p->last_diff_pos = is_diff? i : 0;
	++(q->n_entries);
	++(stack->n_entries);
	if (stack->best > score) stack->best = score;
}

static inline void gap_pop(const bwt_t *bwt, gap_stack_t *stack, gap_entry_t *e)
{
	gap_stack1_t *q;
	q = stack->stacks + stack->best;
	*e = q->stack[q->n_entries - 1];
	--(q->n_entries);
	--(stack->n_entries);
	if (q->n_entries == 0 && stack->n_entries) { // reset best
		int i;
		for (i = stack->best + 1; i < stack->n_stacks; ++i)
			if (stack->stacks[i].n_entries != 0) break;
		stack->best = i;
	} else if (stack->n_entries == 0) stack->best = stack->n_stacks;
	/* PREFETCH the entry that the NEXT gap_pop will take. This is a longer and better-targeted
	 * distance than prefetching at push time: a whole node expansion (one bwt_2occ4 plus up to
	 * nine pushes) separates this from the use. */
	if (stack->n_entries) {
		const gap_stack1_t *nq = stack->stacks + stack->best;
		if (nq->n_entries) {
			const gap_entry_t *n = nq->stack + (nq->n_entries - 1);
			if (n->k) __builtin_prefetch(bwt_occ_intv(bwt, n->k - 1), 0, 3);
			__builtin_prefetch(bwt_occ_intv(bwt, n->l), 0, 3);
		}
	}
}

static inline void gap_shadow(int x, int len, bwtint_t max, int last_diff_pos, bwt_width_t *w)
{
	int i, j;
	for (i = j = 0; i < last_diff_pos; ++i) {
		if (w[i].w > x) w[i].w -= x;
		else if (w[i].w == x) {
			w[i].bid = 1;
			w[i].w = max - (++j);
		} // else should not happen
	}
}

static inline int int_log2(uint32_t v)
{
	int c = 0;
	if (v & 0xffff0000u) { v >>= 16; c |= 16; }
	if (v & 0xff00) { v >>= 8; c |= 8; }
	if (v & 0xf0) { v >>= 4; c |= 4; }
	if (v & 0xc) { v >>= 2; c |= 2; }
	if (v & 0x2) c |= 1;
	return c;
}

/* ---- batched reconcile -------------------------------------------------------------
 * bwt_match_gap is latency-bound on random probes into a multi-GB index: measured thread scaling
 * is 81% efficient out to 16 cores and SMT still buys 1.35x, i.e. the pipeline is full of stall
 * cycles rather than saturating a shared resource. Interleaving several INDEPENDENT reads within
 * one thread lets their cache misses overlap.
 *
 * The search is factored into init + step so the single-read and batched paths execute the SAME
 * body and cannot drift. Interleaving cannot change any individual read's result -- each run owns
 * its stack and state, and steps in the same order -- so the batched path is bit-exact.
 */
typedef struct {
	bwt_t *bwt; int len; const ubyte_t *seq; bwt_width_t *width, *seed_width;
	const gap_opt_t *opt; gap_stack_t *stack;
	int best_score, best_diff, max_diff, best_cnt, max_entries, n_aln, m_aln;
	bwt_aln1_t *aln;
	long long n_pop; int hit_cap;
	int done;
} gap_run_t;

/* one iteration of the former while-loop; 1 = keep stepping, 0 = this read is finished */
static int gap_run_step(gap_run_t *r)
{
	int j;
	if (r->stack->n_entries == 0) return 0;
	(void)j;
	{

		gap_entry_t e;
		int i, m, m_seed = 0, hit_found, allow_diff, allow_M, tmp;
		bwtint_t k, l, cnt_k[4], cnt_l[4], occ;

		if (r->max_entries < r->stack->n_entries) r->max_entries = r->stack->n_entries;
#ifdef ALN_PROFILE
		++r->n_pop;
		if (r->stack->n_entries > r->opt->max_entries) { r->hit_cap = 1; return 0; }
#else
		if (r->stack->n_entries > r->opt->max_entries) return 0;
#endif
		gap_pop(r->bwt, r->stack, &e); // get the best entry
		k = e.k; l = e.l; // SA interval
		i = e.info&0xffff; // length
		if (!(r->opt->mode & BWA_MODE_NONSTOP) && e.info>>21 > r->best_score + r->opt->s_mm) return 0; // no need to proceed

		m = r->max_diff - (e.n_mm + e.n_gapo);
		if (r->opt->mode & BWA_MODE_GAPE) m -= e.n_gape;
		if (m < 0) return 1;
		if (r->seed_width) { // apply seeding
			m_seed = r->opt->max_seed_diff - (e.n_mm + e.n_gapo);
			if (r->opt->mode & BWA_MODE_GAPE) m_seed -= e.n_gape;
		}
		//printf("#1\t[%d,%d,%d,%c]\t[%d,%d,%d]\t[%u,%u]\t[%u,%u]\t%d\n", r->stack->n_entries, a, i, "MID"[e.state], e.n_mm, e.n_gapo, e.n_gape, r->width[i-1].bid, r->width[i-1].w, k, l, e.last_diff_pos);
		if (i > 0 && m < r->width[i-1].bid) return 1;

		// check whether a hit is found
		hit_found = 0;
		if (i == 0) hit_found = 1;
		else if (m == 0 && (e.state == STATE_M || (r->opt->mode&BWA_MODE_GAPE) || e.n_gape == r->opt->max_gape)) { // no diff allowed
			if (bwt_match_exact_alt(r->bwt, i, r->seq, &k, &l)) hit_found = 1;
			else return 1; // no hit, skip
		}

		if (hit_found) { // action for found hits
			int score = aln_score(e.n_mm, e.n_gapo, e.n_gape, r->opt);
			int do_add = 1;
			//printf("#2 hits found: %d:(%u,%u)\n", e.n_mm+e.n_gapo, k, l);
			if (r->n_aln == 0) {
				r->best_score = score;
				r->best_diff = e.n_mm + e.n_gapo;
				if (r->opt->mode & BWA_MODE_GAPE) r->best_diff += e.n_gape;
				if (!(r->opt->mode & BWA_MODE_NONSTOP))
					r->max_diff = (r->best_diff + 1 > r->opt->max_diff)? r->opt->max_diff : r->best_diff + 1; // top2 behaviour
			}
			if (score == r->best_score) r->best_cnt += l - k + 1;
			else if (r->best_cnt > r->opt->max_top2) return 0; // top2b behaviour
			if (e.n_gapo) { // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				for (j = 0; j != r->n_aln; ++j)
					if (r->aln[j].k == k && r->aln[j].l == l) break;
				if (j < r->n_aln) do_add = 0;
			}
			if (do_add) { // append
				bwt_aln1_t *p;
				gap_shadow(l - k + 1, r->len, r->bwt->seq_len, e.last_diff_pos, r->width);
				if (r->n_aln == r->m_aln) {
					r->m_aln <<= 1;
					r->aln = (bwt_aln1_t*)realloc(r->aln, r->m_aln * sizeof(bwt_aln1_t));
					memset(r->aln + r->m_aln/2, 0, r->m_aln/2*sizeof(bwt_aln1_t));
				}
				p = r->aln + r->n_aln;
				p->n_mm = e.n_mm; p->n_gapo = e.n_gapo; p->n_gape = e.n_gape;
				p->n_ins = e.n_ins; p->n_del = e.n_del;
				p->k = k; p->l = l;
				p->score = score;
				//fprintf(stderr, "*** n_mm=%d,n_gapo=%d,n_gape=%d,n_ins=%d,n_del=%d\n", e.n_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del);
				++r->n_aln;
			}
			return 1;
		}

		--i;
		bwt_2occ4(r->bwt, k - 1, l, cnt_k, cnt_l); // retrieve Occ values
		occ = l - k + 1;
		// test whether diff is allowed
		allow_diff = allow_M = 1;
		if (i > 0) {
			int ii = i - (r->len - r->opt->seed_len);
			if (r->width[i-1].bid > m-1) allow_diff = 0;
			else if (r->width[i-1].bid == m-1 && r->width[i].bid == m-1 && r->width[i-1].w == r->width[i].w) allow_M = 0;
			if (r->seed_width && ii > 0) {
				if (r->seed_width[ii-1].bid > m_seed-1) allow_diff = 0;
				else if (r->seed_width[ii-1].bid == m_seed-1 && r->seed_width[ii].bid == m_seed-1
						 && r->seed_width[ii-1].w == r->seed_width[ii].w) allow_M = 0;
			}
		}
		// indels
		tmp = (r->opt->mode & BWA_MODE_LOGGAP)? int_log2(e.n_gape + e.n_gapo)/2+1 : e.n_gapo + e.n_gape;
		if (allow_diff && i >= r->opt->indel_end_skip + tmp && r->len - i >= r->opt->indel_end_skip + tmp) {
			if (e.state == STATE_M) { // gap open
				if (e.n_gapo < r->opt->max_gapo) { // gap open is allowed
					// insertion
					gap_push(r->bwt, r->stack, i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins + 1, e.n_del, STATE_I, 1, r->opt);
					// deletion
					for (j = 0; j != 4; ++j) {
						k = r->bwt->L2[j] + cnt_k[j] + 1;
						l = r->bwt->L2[j] + cnt_l[j];
						if (k <= l) gap_push(r->bwt, r->stack, i + 1, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins, e.n_del + 1, STATE_D, 1, r->opt);
					}
				}
			} else if (e.state == STATE_I) { // extention of an insertion
				if (e.n_gape < r->opt->max_gape) // gap extention is allowed
					gap_push(r->bwt, r->stack, i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins + 1, e.n_del, STATE_I, 1, r->opt);
			} else if (e.state == STATE_D) { // extention of a deletion
				if (e.n_gape < r->opt->max_gape) { // gap extention is allowed
					if (e.n_gape + e.n_gapo < r->max_diff || occ < r->opt->max_del_occ) {
						for (j = 0; j != 4; ++j) {
							k = r->bwt->L2[j] + cnt_k[j] + 1;
							l = r->bwt->L2[j] + cnt_l[j];
							if (k <= l) gap_push(r->bwt, r->stack, i + 1, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins, e.n_del + 1, STATE_D, 1, r->opt);
						}
					}
				}
			}
		}
		// mismatches
		if (allow_diff && allow_M) { // mismatch is allowed
			for (j = 1; j <= 4; ++j) {
				int c = (r->seq[i] + j) & 3;
				int is_mm = (j != 4 || r->seq[i] > 3);
				k = r->bwt->L2[c] + cnt_k[c] + 1;
				l = r->bwt->L2[c] + cnt_l[c];
				if (k <= l) gap_push(r->bwt, r->stack, i, k, l, e.n_mm + is_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, STATE_M, is_mm, r->opt);
			}
		} else if (r->seq[i] < 4) { // try exact match only
			int c = r->seq[i] & 3;
			k = r->bwt->L2[c] + cnt_k[c] + 1;
			l = r->bwt->L2[c] + cnt_l[c];
			if (k <= l) gap_push(r->bwt, r->stack, i, k, l, e.n_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, STATE_M, 0, r->opt);
		}
	
	}
	return 1;
}

/* Prefetch the Occ buckets the next pop will touch. The batch loop issues these for EVERY read
 * before stepping any of them, so the distance to use is a whole round of other reads' work --
 * far longer than anything achievable inside a single dependent chain. */
static inline void gap_run_prefetch(const gap_run_t *r)
{
	const gap_stack_t *s = r->stack;
	const gap_stack1_t *q;
	const gap_entry_t *e;
	if (!s->n_entries) return;
	q = s->stacks + s->best;
	if (!q->n_entries) return;
	e = q->stack + (q->n_entries - 1);
	if (e->k) __builtin_prefetch(bwt_occ_intv(r->bwt, e->k - 1), 0, 3);
	__builtin_prefetch(bwt_occ_intv(r->bwt, e->l), 0, 3);
}

static void gap_run_init(gap_run_t *r, bwt_t *const bwt, int len, const ubyte_t *seq,
                         bwt_width_t *width, bwt_width_t *seed_width, const gap_opt_t *opt,
                         gap_stack_t *stack)
{
	int j, _j;
	r->bwt = bwt; r->len = len; r->seq = seq; r->width = width; r->seed_width = seed_width;
	r->opt = opt; r->stack = stack;
	r->best_score = aln_score(opt->max_diff+1, opt->max_gapo+1, opt->max_gape+1, opt);
	r->best_diff = opt->max_diff + 1; r->max_diff = opt->max_diff;
	r->best_cnt = 0; r->max_entries = 0; r->n_pop = 0; r->hit_cap = 0;
	r->m_aln = 4; r->n_aln = 0;
	r->aln = (bwt_aln1_t*)calloc(r->m_aln, sizeof(bwt_aln1_t));
	r->done = 0;
	for (j = _j = 0; j < len; ++j) if (seq[j] > 3) ++_j;
	if (_j > r->max_diff) { r->done = 1; return; }
	gap_reset_stack(stack);
	gap_push(bwt, stack, len, 0, bwt->seq_len, 0, 0, 0, 0, 0, 0, 0, opt);
}

/* Run nb independent reads interleaved in one thread; each needs its own gap_stack_t.
 * aln_out[i]/n_aln_out[i] receive exactly what bwt_match_gap would have returned for read i. */
void bwt_match_gap_batch(bwt_t *const bwt, int nb, const int *len, const ubyte_t **seq,
                         bwt_width_t **width, bwt_width_t **seed_width, const gap_opt_t **opt,
                         gap_stack_t **stacks, bwt_aln1_t **aln_out, int *n_aln_out)
{
	gap_run_t r[BWT_GAP_BATCH_MAX];
	int i, active = 0;
	if (nb > BWT_GAP_BATCH_MAX) nb = BWT_GAP_BATCH_MAX;
	for (i = 0; i < nb; ++i) {
		gap_run_init(&r[i], bwt, len[i], seq[i], width[i], seed_width ? seed_width[i] : 0, opt[i], stacks[i]);
		if (!r[i].done) ++active;
	}
	while (active) {
		for (i = 0; i < nb; ++i) if (!r[i].done) gap_run_prefetch(&r[i]);
		for (i = 0; i < nb; ++i) if (!r[i].done) {
			if (!gap_run_step(&r[i])) { r[i].done = 1; --active; }
		}
	}
	for (i = 0; i < nb; ++i) { aln_out[i] = r[i].aln; n_aln_out[i] = r[i].n_aln; }
}

bwt_aln1_t *bwt_match_gap(bwt_t *const bwt, int len, const ubyte_t *seq, bwt_width_t *width,
						  bwt_width_t *seed_width, const gap_opt_t *opt, int *_n_aln, gap_stack_t *stack)
{ // $seq is the reverse complement of the input read
	gap_run_t r;
	gap_run_init(&r, bwt, len, seq, width, seed_width, opt, stack);
	while (!r.done) { gap_run_prefetch(&r); if (!gap_run_step(&r)) break; }
	*_n_aln = r.n_aln;
#ifdef ALN_PROFILE
	aln_prof_record(r.max_entries, r.n_pop, r.hit_cap, r.n_aln);
#endif
	return r.aln;
}
