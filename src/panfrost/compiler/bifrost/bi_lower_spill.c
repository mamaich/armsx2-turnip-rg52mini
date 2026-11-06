/*
 * Copyright 2025 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/bitset.h"
#include "bi_debug.h"
#include "compiler.h"

struct lower_spill_ctx {
   bi_context *shader;
   uint32_t *sizes;
   /* If bi_index::memory, tls_loc[idx.value] is valid. */
   uint32_t *tls_loc;
   /* Bytes of spill memory used so far. */
   uint32_t spill_count;
};

static void
lower_memmov(struct lower_spill_ctx* ctx, bi_instr *I, uint32_t tls_base)
{
   assert(I->op == BI_OPCODE_MEMMOV);
   assert(I->nr_srcs == 1 && I->nr_dests == 1);
   assert(I->src[0].memory || I->dest[0].memory);

   bi_builder b = bi_init_builder(ctx->shader, bi_before_instr(I));

   bi_index src = I->src[0];
   bi_index dst = I->dest[0];

   if (I->src[0].memory) {
      unsigned offset = ctx->tls_loc[src.value];
      unsigned bits = ctx->sizes[src.value] * 32;
      assert(bits <= 128);
      bi_load_tl(&b, bits, dst, tls_base + offset);
   } else {
      unsigned offset = ctx->tls_loc[dst.value];
      unsigned bits = ctx->sizes[dst.value] * 32;
      assert(bits <= 128);
      bi_store_tl(&b, bits, src, tls_base + offset);
   }

   bi_remove_instruction(I);
}

/* For vector phis, bi_out_of_ssa expects the width in bi_instr::table. */
static void
lower_phi_size(struct lower_spill_ctx *ctx, bi_instr *I)
{
   assert(I->op == BI_OPCODE_PHI);
   assert(I->nr_dests == 1);

#ifndef NDEBUG
   /* Sanity check. */

   unsigned words = 1;
   words = MAX2(words, ctx->sizes[I->dest[0].value]);

   bi_foreach_ssa_src(I, s)
      words = MAX2(words, ctx->sizes[I->src[s].value]);

   assert(words == ctx->sizes[I->dest[0].value]);
#endif

   I->table = ctx->sizes[I->dest[0].value];
   assert(I->table <= 4);
}

static void
lower_mem_phi(struct lower_spill_ctx *ctx, bi_instr *I, uint32_t tls_base)
{
   assert(I->op == BI_OPCODE_PHI);
   assert(I->nr_dests == 1);

   /* PHIs get lowered in bi_out_of_ssa, which expects memory operands to
    * provide the actual TLS offset as bi_index::value.
    */

   if (I->dest[0].memory) {
      const bi_index dst = I->dest[0];
      I->dest[0].value = tls_base + ctx->tls_loc[dst.value];
   }

   bi_foreach_src(I, s) {
      const bi_index src = I->src[s];
      if (!src.memory)
         continue;

      assert(ctx->tls_loc[src.value] != UINT32_MAX && "Undefined source");
      I->src[s].value = tls_base + ctx->tls_loc[src.value];
   }
}

/*
 * Each memory variable used to get a slot of its own, so a phi spilled to
 * memory was a load and a store on every edge into its block, from the slot
 * of its source to its own. The PICA vertex shaders of Azahar carry dozens
 * of registers around their dispatcher loop that way: 62 of the 223 fills
 * and 126 spills of the varying half of a Dark Moon shader were such copies.
 * Give a phi and its sources one slot where their live ranges allow it, and
 * those copies move a slot onto itself, which bi_out_of_ssa drops.
 *
 * Interference is computed for the memory variables alone. bi_out_of_ssa
 * stores a spilled phi into its slot at the end of each predecessor, one phi
 * after the other, so a phi is taken to be written there: it conflicts with
 * everything live out of the predecessors but its own source on that edge
 * (unless the source lives on past the edge), as well as with what is live
 * after the phis of its block.
 */
#define BI_MEM_COALESCE_MAX 4096

struct mem_coalesce {
   unsigned n, words;
   /* SSA index -> memory variable, or UINT32_MAX */
   uint32_t *id;
   /* Per variable, the variables it interferes with; for the root of a
    * class, the union over the class. */
   BITSET_WORD *interf;
   /* Union-find over the variables; for a root, the members of its class */
   uint32_t *parent;
   BITSET_WORD *members;
   /* TLS offset of a class, by its root, or UINT32_MAX */
   uint32_t *slot;
};

static inline unsigned
mc_id(const struct mem_coalesce *mc, bi_index idx)
{
   if (idx.type != BI_INDEX_NORMAL || !idx.memory)
      return UINT32_MAX;

   return mc->id[idx.value];
}

static unsigned
mc_find(struct mem_coalesce *mc, unsigned x)
{
   while (mc->parent[x] != x) {
      mc->parent[x] = mc->parent[mc->parent[x]];
      x = mc->parent[x];
   }

   return x;
}

/* v interferes with every variable in live but itself. */
static void
mc_interfere(struct mem_coalesce *mc, const BITSET_WORD *live, unsigned v)
{
   BITSET_WORD *row = &mc->interf[v * mc->words];
   unsigned i;

   for (unsigned w = 0; w < mc->words; ++w)
      row[w] |= live[w];

   BITSET_FOREACH_SET(i, live, mc->n)
      BITSET_SET(&mc->interf[i * mc->words], v);

   BITSET_CLEAR(row, v);
}

/* What is live at the end of block: live into its successors, and the
 * sources of their phis along the edge from block. */
static void
mc_live_out(struct mem_coalesce *mc, const BITSET_WORD *live_in,
            bi_block *block, BITSET_WORD *out)
{
   memset(out, 0, mc->words * sizeof(BITSET_WORD));

   bi_foreach_successor(block, succ) {
      const BITSET_WORD *in = &live_in[succ->index * mc->words];
      unsigned s = bi_predecessor_index(succ, block);

      for (unsigned w = 0; w < mc->words; ++w)
         out[w] |= in[w];

      bi_foreach_phi_in_block(succ, phi) {
         unsigned v = mc_id(mc, phi->src[s]);

         if (v != UINT32_MAX)
            BITSET_SET(out, v);
      }
   }
}

/* The phis of the successors of block are written at its end, after
 * everything live out of it was computed, own sources aside. */
static void
mc_interfere_succ_phis(struct mem_coalesce *mc, const BITSET_WORD *live_in,
                       bi_block *block, const BITSET_WORD *live_out,
                       BITSET_WORD *tmp)
{
   bi_foreach_successor(block, succ) {
      const BITSET_WORD *in = &live_in[succ->index * mc->words];
      unsigned s = bi_predecessor_index(succ, block);

      bi_foreach_phi_in_block(succ, phi) {
         unsigned p = mc_id(mc, phi->dest[0]);
         if (p == UINT32_MAX)
            continue;

         memcpy(tmp, live_out, mc->words * sizeof(BITSET_WORD));

         unsigned v = mc_id(mc, phi->src[s]);
         if (v != UINT32_MAX && !BITSET_TEST(in, v))
            BITSET_CLEAR(tmp, v);

         mc_interfere(mc, tmp, p);
      }
   }
}

static void
mc_union(struct mem_coalesce *mc, unsigned a, unsigned b)
{
   a = mc_find(mc, a);
   b = mc_find(mc, b);

   if (a == b)
      return;

   BITSET_WORD *ia = &mc->interf[a * mc->words];
   BITSET_WORD *ib = &mc->interf[b * mc->words];
   BITSET_WORD *ma = &mc->members[a * mc->words];
   BITSET_WORD *mb = &mc->members[b * mc->words];

   for (unsigned w = 0; w < mc->words; ++w) {
      if (ia[w] & mb[w])
         return;
   }

   for (unsigned w = 0; w < mc->words; ++w) {
      ia[w] |= ib[w];
      ma[w] |= mb[w];
   }

   mc->parent[b] = a;
}

/* Returns false, leaving every variable a slot of its own, when there are
 * too many memory variables for the bit matrices. */
static bool
mem_coalesce_init(struct mem_coalesce *mc, bi_context *ctx, uint32_t *sizes,
                  void *memctx)
{
   mc->id = ralloc_array(memctx, uint32_t, ctx->ssa_alloc);
   memset(mc->id, 0xff, sizeof(uint32_t) * ctx->ssa_alloc);
   mc->n = 0;

   bi_foreach_instr_global(ctx, I) {
      bi_foreach_ssa_dest(I, d) {
         if (I->dest[d].memory && mc->id[I->dest[d].value] == UINT32_MAX)
            mc->id[I->dest[d].value] = mc->n++;
      }
   }

   if (mc->n == 0 || mc->n > BI_MEM_COALESCE_MAX)
      return false;

   const unsigned n = mc->n;
   const unsigned words = BITSET_WORDS(n);
   const size_t row_bytes = words * sizeof(BITSET_WORD);
   mc->words = words;
   mc->interf = rzalloc_array(memctx, BITSET_WORD, (size_t)n * words);
   mc->members = rzalloc_array(memctx, BITSET_WORD, (size_t)n * words);
   mc->parent = ralloc_array(memctx, uint32_t, n);
   mc->slot = ralloc_array(memctx, uint32_t, n);
   memset(mc->slot, 0xff, sizeof(uint32_t) * n);

   for (unsigned i = 0; i < n; ++i) {
      mc->parent[i] = i;
      BITSET_SET(&mc->members[i * words], i);
   }

   /* Liveness of the memory variables, by blocks */
   BITSET_WORD *live_in =
      rzalloc_array(memctx, BITSET_WORD, (size_t)ctx->num_blocks * words);
   BITSET_WORD *live = ralloc_array(memctx, BITSET_WORD, words);
   BITSET_WORD *tmp = ralloc_array(memctx, BITSET_WORD, words);
   bool progress;

   do {
      progress = false;

      bi_foreach_block_rev(ctx, block) {
         mc_live_out(mc, live_in, block, live);

         bi_foreach_instr_in_block_rev(block, I) {
            bi_foreach_ssa_dest(I, d) {
               unsigned v = mc_id(mc, I->dest[d]);
               if (v != UINT32_MAX)
                  BITSET_CLEAR(live, v);
            }

            /* Phi sources are read in the predecessors. */
            if (I->op == BI_OPCODE_PHI)
               continue;

            bi_foreach_ssa_src(I, s) {
               unsigned v = mc_id(mc, I->src[s]);
               if (v != UINT32_MAX)
                  BITSET_SET(live, v);
            }
         }

         BITSET_WORD *in = &live_in[block->index * words];
         if (memcmp(in, live, row_bytes)) {
            memcpy(in, live, row_bytes);
            progress = true;
         }
      }
   } while (progress);

   /* Interference */
   bi_foreach_block(ctx, block) {
      mc_live_out(mc, live_in, block, live);
      mc_interfere_succ_phis(mc, live_in, block, live, tmp);

      bi_foreach_instr_in_block_rev(block, I) {
         if (I->op == BI_OPCODE_PHI)
            break;

         bi_foreach_ssa_dest(I, d) {
            unsigned v = mc_id(mc, I->dest[d]);
            if (v != UINT32_MAX) {
               mc_interfere(mc, live, v);
               BITSET_CLEAR(live, v);
            }
         }

         bi_foreach_ssa_src(I, s) {
            unsigned v = mc_id(mc, I->src[s]);
            if (v != UINT32_MAX)
               BITSET_SET(live, v);
         }
      }

      bi_foreach_phi_in_block(block, phi) {
         unsigned v = mc_id(mc, phi->dest[0]);
         if (v != UINT32_MAX)
            mc_interfere(mc, live, v);
      }
   }

   /* Coalesce each spilled phi with its sources */
   bi_foreach_block(ctx, block) {
      bi_foreach_phi_in_block(block, phi) {
         unsigned p = mc_id(mc, phi->dest[0]);
         if (p == UINT32_MAX)
            continue;

         bi_foreach_src(phi, s) {
            unsigned v = mc_id(mc, phi->src[s]);

            if (v != UINT32_MAX &&
                sizes[phi->src[s].value] == sizes[phi->dest[0].value])
               mc_union(mc, p, v);
         }
      }
   }

   return true;
}

static void
assign_tls_locations(struct lower_spill_ctx *ctx, struct mem_coalesce *mc) {
   bi_foreach_instr_global(ctx->shader, I) {
      bi_foreach_ssa_dest(I, d) {
         bi_index dst = I->dest[d];
         if (!dst.memory)
            continue;

         assert(I->op == BI_OPCODE_MEMMOV || I->op == BI_OPCODE_PHI);
         assert(ctx->tls_loc[dst.value] == UINT32_MAX && "Broken SSA");

         unsigned root = UINT32_MAX;
         if (mc) {
            root = mc_find(mc, mc->id[dst.value]);

            if (mc->slot[root] != UINT32_MAX) {
               ctx->tls_loc[dst.value] = mc->slot[root];
               continue;
            }
         }

         unsigned size = ctx->sizes[dst.value];
         unsigned alignment = (size <= 1) ? 4 : (size == 2) ? 8 : 16;

         ctx->spill_count = ALIGN_POT(ctx->spill_count, alignment);
         ctx->tls_loc[dst.value] = ctx->spill_count;
         ctx->spill_count += 4 * size;

         if (mc)
            mc->slot[root] = ctx->tls_loc[dst.value];
      }
   }
}

unsigned
bi_lower_spill(bi_context* ctx, uint32_t tls_base) {
   void* memctx = ralloc_context(NULL);

   uint32_t *sizes = rzalloc_array(memctx, uint32_t, ctx->ssa_alloc);
   uint32_t *tls_loc = ralloc_array(memctx, uint32_t, ctx->ssa_alloc);
   memset(tls_loc, 0xff, sizeof(uint32_t) * ctx->ssa_alloc);

   bi_record_sizes(ctx, sizes);

   struct lower_spill_ctx lctx = {
      .shader = ctx,
      .sizes = sizes,
      .tls_loc = tls_loc,
      .spill_count = 0,
   };

   struct mem_coalesce mc;
   bool coalesce = !(bifrost_debug & BIFROST_DBG_NOMEMCOALESCE) &&
                   mem_coalesce_init(&mc, ctx, sizes, memctx);

   assign_tls_locations(&lctx, coalesce ? &mc : NULL);

   /* Before adding SSA values, lower phi sizes. */
   bi_foreach_instr_global_safe(ctx, I) {
      if (I->op == BI_OPCODE_PHI)
         lower_phi_size(&lctx, I);
   }

   bi_foreach_instr_global_safe(ctx, I) {
      switch (I->op) {
      case BI_OPCODE_MEMMOV:
         lower_memmov(&lctx, I, tls_base);
         break;
      case BI_OPCODE_PHI:
         lower_mem_phi(&lctx, I, tls_base);
         break;
      default:
         break;
      }
   }

   ralloc_free(memctx);

   return lctx.spill_count;
}
