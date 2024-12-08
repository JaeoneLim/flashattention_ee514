
#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "flash.h"
#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

template <typename Ktraits, bool Is_causal, bool Is_local, typename TileScheduler, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps * 32, 1)
compute_attn_a100(typename CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>::Params const mainloop_params,
                  typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>::Params const epilogue_params,
                  typename TileScheduler::Params const scheduler_params,
                  Seqlen_traits_Q seqlen_traits_q, 
                  Seqlen_traits seqlen_traits_k) {
    
    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);
    static constexpr bool Is_WS = Ktraits::Is_WS;
    static constexpr bool No_smem_O = Ktraits::No_smem_O;

    // A100-specific constants
    static constexpr int NumThreadsPerWarp = 32;
    static constexpr int NumThreadsPerBlock = Ktraits::kNWarps * NumThreadsPerWarp;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockH = Ktraits::kBlockH;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    const int warp_idx = threadIdx.x / NumThreadsPerWarp;
    const int lane_idx = threadIdx.x % NumThreadsPerWarp;
    const bool is_leader = lane_idx == 0;

    // Initialize shared memory barriers
    if (threadIdx.x == 0) {
        shared_storage.init_barriers();
    }
    __syncthreads();

    // Compute grid coordinates
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    // Initialize scheduler
    TileScheduler scheduler(&shared_storage.tile_scheduler);
    
    // Producer warp handles loading Q, K, V
    if (warp_idx == 0) {
        int work_idx = 0;
        
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            // Initialize sequence length trackers
            seqlen_traits_q.init(bidb);
            seqlen_traits_k.init(bidb);

            // Check sequence length boundary
            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }

            // Calculate n_block bounds
            int n_block_min = 0, n_block_max;
            get_n_block_min_max(mainloop_params, m_block, n_split_idx, 
                              seqlen_traits_q, seqlen_traits_k,
                              n_block_min, n_block_max);

            // Skip if no valid blocks
            if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if (n_block_max <= n_block_min) {
                    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
                    scheduler.broadcast_next_work(work_tile_info);
                    continue;
                }
            }

            // Load Q, K, V tiles using LDG
            load_qkv_tiles(mainloop_params, shared_storage, work_tile_info, 
                          block_coord, work_idx, n_block_min, n_block_max);
            
            ++work_idx;
        }

        // Signal completion of loading
        shared_storage.producer_done.store(1, cuda::memory_order_release);
    }
    // Consumer warps handle computation
    else {
        // Initialize matmul state
        typename Ktraits::TiledMma tiled_mma;
        typename Ktraits::TiledAccumulator tiled_acc;

        int work_idx = 0;
        scheduler.init_consumer();

        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {

            // Set up accumulator for attention output
            Tensor acc = partition_fragment_C(tiled_acc, TileShape_MNK{});
            flash::Softmax<2 * (2 * kBlockM / (NumThreadsPerBlock - NumThreadsPerWarp))> 
                softmax(mainloop_params.softmax_scale_log2);

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            // Initialize sequence lengths
            seqlen_traits_q.init(bidb);
            seqlen_traits_k.init(bidb);

            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }

            // Calculate n_block bounds
            int n_block_min = 0, n_block_max;
            get_n_block_min_max(mainloop_params, m_block, n_split_idx,
                              seqlen_traits_q, seqlen_traits_k, 
                              n_block_min, n_block_max);

            if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if (n_block_max <= n_block_min) {
                    store_zero_output(epilogue_params, shared_storage, block_coord, 
                                   seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);
                    continue;
                }
            }

            // Compute attention scores and output
            compute_attention(mainloop_params, shared_storage, acc, softmax,
                            n_block_min, n_block_max, work_idx, m_block,
                            seqlen_traits_q, seqlen_traits_k);

            // Store results using STG
            store_results(epilogue_params, acc, softmax.row_sum, shared_storage,
                         block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);

            ++work_idx;
        }
    }
}

} // namespace flash