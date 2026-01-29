"""
Flash-KMeans Assignment Kernel - FA2 Style

Key improvements over FA1:
1. Split-K parallelization: Parallelize over the centroid dimension K
2. Two-pass algorithm: Partial results + reduction
3. Better occupancy through smaller per-block work
4. Pipelined memory access patterns

The FA2 approach adds a third grid dimension to parallelize over K-splits,
similar to how Flash Attention 2 parallelizes over sequence length.
"""

import torch
import triton
import triton.language as tl


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# =============================================================================
# FA2-Style Kernel: Split-K with Two-Pass Reduction
# =============================================================================

# -----------------------------------------------------------------------------
# Pass 1: Compute partial min/argmin for each K-split
# -----------------------------------------------------------------------------

_TUNE_CONFIGS_FA2_PASS1 = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=ns, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [64, 128, 256]
    for wp in [4, 8]
    for ns in [2, 3, 4]
]


def _cfg_keep_fa2(conf):
    """Prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    if BN * BK < 32 * 64 and conf.num_warps > 4:
        return False
    return True


_TUNE_CONFIGS_FA2_PASS1 = list(filter(_cfg_keep_fa2, _TUNE_CONFIGS_FA2_PASS1))


@triton.autotune(_TUNE_CONFIGS_FA2_PASS1, key=["N", "K", "NUM_K_SPLITS"])
@triton.jit
def _euclid_assign_fa2_pass1_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    x_sq_ptr,              # *f32         [B, N]
    c_sq_ptr,              # *f32         [B, K]
    partial_dist_ptr,      # *f32         [B, NUM_K_SPLITS, N] - partial min distances
    partial_idx_ptr,       # *i32         [B, NUM_K_SPLITS, N] - partial argmin indices
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_pdist_b: tl.constexpr,
    stride_pdist_s: tl.constexpr,
    stride_pdist_n: tl.constexpr,
    stride_pidx_b: tl.constexpr,
    stride_pidx_s: tl.constexpr,
    stride_pidx_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FA2-style Pass 1: Each program handles a tile of BLOCK_N points 
    for a SUBSET of centroids (one K-split).
    
    Grid: (ceil(N/BLOCK_N), NUM_K_SPLITS, B)
    
    This parallelizes over K, unlike FA1 which serializes over K.
    """
    pid_n = tl.program_id(0)          # tile index along N dimension
    pid_k_split = tl.program_id(1)    # which K-split this program handles
    pid_b = tl.program_id(2)          # batch index

    # Compute K range for this split
    k_per_split = tl.cdiv(K, NUM_K_SPLITS)
    k_start_split = pid_k_split * k_per_split
    k_end_split = tl.minimum(k_start_split + k_per_split, K)
    
    # Early exit if this split has no work
    if k_start_split >= K:
        return

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile (BLOCK_N, D) - loaded once per program
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)

    # Pre-load x_sq for the tile (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index for THIS K-SPLIT
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in THIS SPLIT only (key FA2 difference!)
    # ------------------------------------------------------------------
    for k_start in range(k_start_split, k_end_split, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < k_end_split

        # Load centroid tile (D, BLOCK_K)
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)

        # Load c_sq for this centroid tile (BLOCK_K,)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)

        # Squared Euclidean distance: ||x||^2 + ||c||^2 - 2*x.c
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)

        # Mask out invalid centroid columns
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write partial results for this K-split
    # ------------------------------------------------------------------
    pdist_ptrs = (
        partial_dist_ptr
        + pid_b * stride_pdist_b
        + pid_k_split * stride_pdist_s
        + n_offsets * stride_pdist_n
    )
    pidx_ptrs = (
        partial_idx_ptr
        + pid_b * stride_pidx_b
        + pid_k_split * stride_pidx_s
        + n_offsets * stride_pidx_n
    )
    tl.store(pdist_ptrs, best_dist, mask=n_mask)
    tl.store(pidx_ptrs, best_idx, mask=n_mask)


# -----------------------------------------------------------------------------
# Pass 2: Reduce partial results across K-splits
# -----------------------------------------------------------------------------

_TUNE_CONFIGS_FA2_PASS2 = [
    triton.Config({"BLOCK_N": BN}, num_stages=ns, num_warps=wp)
    for BN in [128, 256, 512]
    for wp in [4, 8]
    for ns in [1, 2]
]


@triton.autotune(_TUNE_CONFIGS_FA2_PASS2, key=["N", "NUM_K_SPLITS"])
@triton.jit
def _euclid_assign_fa2_pass2_kernel(
    partial_dist_ptr,      # *f32 [B, NUM_K_SPLITS, N]
    partial_idx_ptr,       # *i32 [B, NUM_K_SPLITS, N]
    out_ptr,               # *i32 [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    stride_pdist_b: tl.constexpr,
    stride_pdist_s: tl.constexpr,
    stride_pdist_n: tl.constexpr,
    stride_pidx_b: tl.constexpr,
    stride_pidx_s: tl.constexpr,
    stride_pidx_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FA2-style Pass 2: Reduce partial min/argmin across K-splits.
    
    Grid: (ceil(N/BLOCK_N), B)
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Init final best
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # Reduce across all K-splits
    for s in range(NUM_K_SPLITS):
        pdist_ptrs = (
            partial_dist_ptr
            + pid_b * stride_pdist_b
            + s * stride_pdist_s
            + n_offsets * stride_pdist_n
        )
        pidx_ptrs = (
            partial_idx_ptr
            + pid_b * stride_pidx_b
            + s * stride_pidx_s
            + n_offsets * stride_pidx_n
        )
        
        curr_dist = tl.load(pdist_ptrs, mask=n_mask, other=3.4e38)
        curr_idx = tl.load(pidx_ptrs, mask=n_mask, other=0)

        update = curr_dist < best_dist
        best_dist = tl.where(update, curr_dist, best_dist)
        best_idx = tl.where(update, curr_idx, best_idx)

    # Write final result
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


# =============================================================================
# FA2 Alternative: Fused Single-Pass with Better Tiling (for moderate K)
# =============================================================================

_TUNE_CONFIGS_FA2_FUSED = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_D": BD}, num_stages=ns, num_warps=wp)
    for BN in [32, 64]
    for BK in [32, 64, 128]
    for BD in [32, 64, 128]
    for wp in [4, 8]
    for ns in [2, 3, 4]
    if BN * BK >= 32 * 32  # ensure enough work per block
]


def _cfg_keep_fused(conf):
    """Prune configs for the fused kernel."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    BD = conf.kwargs["BLOCK_D"]
    # Avoid too small or unbalanced configs
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    # BD should be reasonable
    if BD < 32:
        return False
    return True


_TUNE_CONFIGS_FA2_FUSED = list(filter(_cfg_keep_fused, _TUNE_CONFIGS_FA2_FUSED))


@triton.autotune(_TUNE_CONFIGS_FA2_FUSED, key=["N", "K", "D"])
@triton.jit
def _euclid_assign_fa2_fused_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    x_sq_ptr,              # *f32         [B, N]
    c_sq_ptr,              # *f32         [B, K]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FA2-style fused kernel with D-dimension tiling.
    
    Key FA2 improvements:
    1. Tile over D dimension for large embedding dimensions
    2. Accumulate partial dot products
    3. Better register utilization
    
    This is beneficial when D is large and doesn't fit well in registers.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Pre-load x_sq (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # Iterate over centroids
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load c_sq (BLOCK_K,)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Accumulate dot product over D tiles
        cross = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
        
        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D

            # Load x tile (BLOCK_N, BLOCK_D)
            x_ptrs = (
                x_ptr
                + pid_b * stride_x_b
                + n_offsets[:, None] * stride_x_n
                + d_offsets[None, :] * stride_x_d
            )
            x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            # Load c tile (BLOCK_D, BLOCK_K)
            c_ptrs = (
                c_ptr
                + pid_b * stride_c_b
                + k_offsets[None, :] * stride_c_k
                + d_offsets[:, None] * stride_c_d
            )
            c_tile = tl.load(c_ptrs, mask=d_mask[:, None] & k_mask[None, :], other=0.0)

            # Accumulate partial dot product
            cross += tl.dot(x_tile, c_tile).to(tl.float32)

        # Compute squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # Write results
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def euclid_assign_triton_fa2_split_k(
    x: torch.Tensor, 
    centroids: torch.Tensor, 
    x_sq: torch.Tensor, 
    out: torch.Tensor = None, 
    c_sq: torch.Tensor = None,
    num_k_splits: int = None,
) -> torch.Tensor:
    """
    FA2-style nearest-centroid assignment with Split-K parallelization.
    
    This uses a two-pass algorithm:
    1. Pass 1: Each K-split computes partial min/argmin in parallel
    2. Pass 2: Reduce partial results across K-splits
    
    Args:
        x         : (B, N, D) float16 / float32
        centroids : (B, K, D) same dtype as x
        x_sq      : (B, N)    float32 - ||x||^2 per point
        out       : (B, N)    int32   - (optional) pre-allocated output
        c_sq      : (B, K)    float32 - (optional) ||centroids||^2
        num_k_splits: int     - number of K-splits (auto-selected if None)
    
    Returns:
        cluster_ids (B, N) int32
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda
    assert centroids.dtype == x.dtype

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D)
    assert x_sq.shape == (B, N)

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    if c_sq is None:
        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

    # Auto-select number of K-splits based on K
    # More splits = more parallelism but more memory for partial results
    if num_k_splits is None:
        if K <= 128:
            num_k_splits = 1  # Small K, no benefit from splitting
        elif K <= 512:
            num_k_splits = 2
        elif K <= 2048:
            num_k_splits = 4
        else:
            num_k_splits = 8

    # If only 1 split, fall back to fused kernel (no overhead)
    if num_k_splits == 1:
        return euclid_assign_triton_fa2_fused(x, centroids, x_sq, out, c_sq)

    # Allocate partial results buffers
    partial_dist = torch.empty((B, num_k_splits, N), device=x.device, dtype=torch.float32)
    partial_idx = torch.empty((B, num_k_splits, N), device=x.device, dtype=torch.int32)

    # Strides
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_pdist_b, stride_pdist_s, stride_pdist_n = partial_dist.stride()
    stride_pidx_b, stride_pidx_s, stride_pidx_n = partial_idx.stride()
    stride_out_b, stride_out_n = out.stride()

    # Pass 1: Compute partial results
    grid_pass1 = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        num_k_splits,
        B,
    )

    _euclid_assign_fa2_pass1_kernel[grid_pass1](
        x, centroids, x_sq, c_sq,
        partial_dist, partial_idx,
        B, N, K, D, num_k_splits,
        stride_x_b, stride_x_n, stride_x_d,
        stride_c_b, stride_c_k, stride_c_d,
        stride_xsq_b, stride_xsq_n,
        stride_csq_b, stride_csq_k,
        stride_pdist_b, stride_pdist_s, stride_pdist_n,
        stride_pidx_b, stride_pidx_s, stride_pidx_n,
    )

    # Pass 2: Reduce across K-splits
    grid_pass2 = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _euclid_assign_fa2_pass2_kernel[grid_pass2](
        partial_dist, partial_idx, out,
        B, N, num_k_splits,
        stride_pdist_b, stride_pdist_s, stride_pdist_n,
        stride_pidx_b, stride_pidx_s, stride_pidx_n,
        stride_out_b, stride_out_n,
    )

    return out


def euclid_assign_triton_fa2_fused(
    x: torch.Tensor, 
    centroids: torch.Tensor, 
    x_sq: torch.Tensor, 
    out: torch.Tensor = None, 
    c_sq: torch.Tensor = None,
) -> torch.Tensor:
    """
    FA2-style nearest-centroid assignment with D-dimension tiling.
    
    This is a single-pass fused kernel that tiles over the D dimension,
    beneficial when D is large.
    
    Args:
        x         : (B, N, D) float16 / float32
        centroids : (B, K, D) same dtype as x
        x_sq      : (B, N)    float32 - ||x||^2 per point
        out       : (B, N)    int32   - (optional) pre-allocated output
        c_sq      : (B, K)    float32 - (optional) ||centroids||^2
    
    Returns:
        cluster_ids (B, N) int32
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda
    assert centroids.dtype == x.dtype

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D)
    assert x_sq.shape == (B, N)

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    if c_sq is None:
        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _euclid_assign_fa2_fused_kernel[grid](
        x, centroids, x_sq, c_sq, out,
        B, N, K, D,
        stride_x_b, stride_x_n, stride_x_d,
        stride_c_b, stride_c_k, stride_c_d,
        stride_xsq_b, stride_xsq_n,
        stride_csq_b, stride_csq_k,
        stride_out_b, stride_out_n,
    )
    return out


# =============================================================================
# Cosine Similarity Variants (FA2-style)
# =============================================================================

@triton.autotune(_TUNE_CONFIGS_FA2_PASS1, key=["N", "K", "NUM_K_SPLITS"])
@triton.jit
def _cosine_assign_fa2_pass1_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D] (normalized)
    c_ptr,                 # *f16 / *f32 [B, K, D] (normalized)
    partial_sim_ptr,       # *f32         [B, NUM_K_SPLITS, N] - partial max similarities
    partial_idx_ptr,       # *i32         [B, NUM_K_SPLITS, N] - partial argmax indices
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_psim_b: tl.constexpr,
    stride_psim_s: tl.constexpr,
    stride_psim_n: tl.constexpr,
    stride_pidx_b: tl.constexpr,
    stride_pidx_s: tl.constexpr,
    stride_pidx_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FA2-style Pass 1 for cosine similarity (maximize dot product)."""
    pid_n = tl.program_id(0)
    pid_k_split = tl.program_id(1)
    pid_b = tl.program_id(2)

    k_per_split = tl.cdiv(K, NUM_K_SPLITS)
    k_start_split = pid_k_split * k_per_split
    k_end_split = tl.minimum(k_start_split + k_per_split, K)
    
    if k_start_split >= K:
        return

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    offs_d = tl.arange(0, D)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)

    # For cosine, we want max similarity (not min distance)
    best_sim = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in range(k_start_split, k_end_split, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < k_end_split

        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)

        # Cosine similarity = dot product (assuming normalized)
        sim = tl.dot(x_tile, c_tile).to(tl.float32)
        sim = tl.where(k_mask[None, :], sim, -3.4e38)

        curr_max = tl.max(sim, axis=1)
        curr_idx = tl.argmax(sim, axis=1)

        update = curr_max > best_sim
        best_sim = tl.where(update, curr_max, best_sim)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    psim_ptrs = (
        partial_sim_ptr
        + pid_b * stride_psim_b
        + pid_k_split * stride_psim_s
        + n_offsets * stride_psim_n
    )
    pidx_ptrs = (
        partial_idx_ptr
        + pid_b * stride_pidx_b
        + pid_k_split * stride_pidx_s
        + n_offsets * stride_pidx_n
    )
    tl.store(psim_ptrs, best_sim, mask=n_mask)
    tl.store(pidx_ptrs, best_idx, mask=n_mask)


@triton.autotune(_TUNE_CONFIGS_FA2_PASS2, key=["N", "NUM_K_SPLITS"])
@triton.jit
def _cosine_assign_fa2_pass2_kernel(
    partial_sim_ptr,       # *f32 [B, NUM_K_SPLITS, N]
    partial_idx_ptr,       # *i32 [B, NUM_K_SPLITS, N]
    out_ptr,               # *i32 [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    stride_psim_b: tl.constexpr,
    stride_psim_s: tl.constexpr,
    stride_psim_n: tl.constexpr,
    stride_pidx_b: tl.constexpr,
    stride_pidx_s: tl.constexpr,
    stride_pidx_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FA2-style Pass 2 for cosine (reduce max across splits)."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    best_sim = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for s in range(NUM_K_SPLITS):
        psim_ptrs = (
            partial_sim_ptr
            + pid_b * stride_psim_b
            + s * stride_psim_s
            + n_offsets * stride_psim_n
        )
        pidx_ptrs = (
            partial_idx_ptr
            + pid_b * stride_pidx_b
            + s * stride_pidx_s
            + n_offsets * stride_pidx_n
        )
        
        curr_sim = tl.load(psim_ptrs, mask=n_mask, other=-3.4e38)
        curr_idx = tl.load(pidx_ptrs, mask=n_mask, other=0)

        update = curr_sim > best_sim
        best_sim = tl.where(update, curr_sim, best_sim)
        best_idx = tl.where(update, curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


def cosine_assign_triton_fa2_split_k(
    x: torch.Tensor, 
    centroids: torch.Tensor, 
    out: torch.Tensor = None,
    num_k_splits: int = None,
) -> torch.Tensor:
    """
    FA2-style cosine similarity assignment with Split-K parallelization.
    
    Args:
        x         : (B, N, D) float16 / float32 (normalized)
        centroids : (B, K, D) same dtype as x (normalized)
        out       : (B, N)    int32 - (optional) pre-allocated output
        num_k_splits: int     - number of K-splits (auto-selected if None)
    
    Returns:
        cluster_ids (B, N) int32
    """
    assert x.is_cuda and centroids.is_cuda
    assert centroids.dtype == x.dtype

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D)

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    # Auto-select number of K-splits
    if num_k_splits is None:
        if K <= 128:
            num_k_splits = 1
        elif K <= 512:
            num_k_splits = 2
        elif K <= 2048:
            num_k_splits = 4
        else:
            num_k_splits = 8

    if num_k_splits == 1:
        # Fall back to original FA1 kernel for small K
        from flash_kmeans.assign_euclid_triton import cosine_assign_triton
        return cosine_assign_triton(x, centroids, out)

    partial_sim = torch.empty((B, num_k_splits, N), device=x.device, dtype=torch.float32)
    partial_idx = torch.empty((B, num_k_splits, N), device=x.device, dtype=torch.int32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_psim_b, stride_psim_s, stride_psim_n = partial_sim.stride()
    stride_pidx_b, stride_pidx_s, stride_pidx_n = partial_idx.stride()
    stride_out_b, stride_out_n = out.stride()

    grid_pass1 = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        num_k_splits,
        B,
    )

    _cosine_assign_fa2_pass1_kernel[grid_pass1](
        x, centroids,
        partial_sim, partial_idx,
        B, N, K, D, num_k_splits,
        stride_x_b, stride_x_n, stride_x_d,
        stride_c_b, stride_c_k, stride_c_d,
        stride_psim_b, stride_psim_s, stride_psim_n,
        stride_pidx_b, stride_pidx_s, stride_pidx_n,
    )

    grid_pass2 = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _cosine_assign_fa2_pass2_kernel[grid_pass2](
        partial_sim, partial_idx, out,
        B, N, num_k_splits,
        stride_psim_b, stride_psim_s, stride_psim_n,
        stride_pidx_b, stride_pidx_s, stride_pidx_n,
        stride_out_b, stride_out_n,
    )

    return out


# =============================================================================
# Convenience aliases
# =============================================================================

# Default FA2 implementations
euclid_assign_triton_fa2 = euclid_assign_triton_fa2_split_k
cosine_assign_triton_fa2 = cosine_assign_triton_fa2_split_k


# =============================================================================
# Quick correctness & performance check
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, D = 32, 74256, 128
    K = 1000
    dtype = torch.float16

    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    cent = torch.randn(B, K, D, device="cuda", dtype=dtype)
    x_sq = (x.to(torch.float32) ** 2).sum(-1)
    c_sq = (cent.to(torch.float32) ** 2).sum(-1)

    # Reference (PyTorch)
    dist = (
        x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum("bnd,bkd->bnk", x, cent).to(torch.float32)
    ).clamp_min_(0.0)
    ref_ids = dist.argmin(dim=-1)

    # FA1 (original)
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    fa1_ids = euclid_assign_triton(x, cent, x_sq, c_sq=c_sq)

    # FA2 Split-K
    fa2_split_k_ids = euclid_assign_triton_fa2_split_k(x, cent, x_sq, c_sq=c_sq, num_k_splits=4)
    
    # FA2 Fused
    fa2_fused_ids = euclid_assign_triton_fa2_fused(x, cent, x_sq, c_sq=c_sq)

    print("=== Correctness Check ===")
    print(f"FA1 vs Reference: {torch.equal(ref_ids.cpu(), fa1_ids.cpu())}")
    print(f"FA2 Split-K vs Reference: {torch.equal(ref_ids.cpu(), fa2_split_k_ids.cpu())}")
    print(f"FA2 Fused vs Reference: {torch.equal(ref_ids.cpu(), fa2_fused_ids.cpu())}")

    # Performance comparison
    repeats = 20
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)

    # Warm-up
    for _ in range(3):
        euclid_assign_triton(x, cent, x_sq, out, c_sq)
        euclid_assign_triton_fa2_split_k(x, cent, x_sq, out, c_sq, num_k_splits=4)
        euclid_assign_triton_fa2_fused(x, cent, x_sq, out, c_sq)

    torch.cuda.synchronize()

    # FA1 timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        euclid_assign_triton(x, cent, x_sq, out, c_sq)
    end.record()
    torch.cuda.synchronize()
    fa1_time = start.elapsed_time(end) / repeats

    # FA2 Split-K timing
    start.record()
    for _ in range(repeats):
        euclid_assign_triton_fa2_split_k(x, cent, x_sq, out, c_sq, num_k_splits=4)
    end.record()
    torch.cuda.synchronize()
    fa2_split_k_time = start.elapsed_time(end) / repeats

    # FA2 Fused timing
    start.record()
    for _ in range(repeats):
        euclid_assign_triton_fa2_fused(x, cent, x_sq, out, c_sq)
    end.record()
    torch.cuda.synchronize()
    fa2_fused_time = start.elapsed_time(end) / repeats

    print(f"\n=== Performance ({B}x{N} points, K={K}, D={D}) ===")
    print(f"FA1 (original):     {fa1_time:.3f} ms")
    print(f"FA2 Split-K:        {fa2_split_k_time:.3f} ms (speedup: {fa1_time/fa2_split_k_time:.2f}x)")
    print(f"FA2 Fused:          {fa2_fused_time:.3f} ms (speedup: {fa1_time/fa2_fused_time:.2f}x)")
    
    # Test with larger K
    print("\n=== Testing with larger K ===")
    for test_K in [2000, 4000, 8000]:
        cent_large = torch.randn(B, test_K, D, device="cuda", dtype=dtype)
        c_sq_large = (cent_large.to(torch.float32) ** 2).sum(-1)
        out_large = torch.empty((B, N), device="cuda", dtype=torch.int32)
        
        # Warm-up
        euclid_assign_triton(x, cent_large, x_sq, out_large, c_sq_large)
        euclid_assign_triton_fa2_split_k(x, cent_large, x_sq, out_large, c_sq_large)
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(repeats):
            euclid_assign_triton(x, cent_large, x_sq, out_large, c_sq_large)
        end.record()
        torch.cuda.synchronize()
        fa1_t = start.elapsed_time(end) / repeats
        
        start.record()
        for _ in range(repeats):
            euclid_assign_triton_fa2_split_k(x, cent_large, x_sq, out_large, c_sq_large)
        end.record()
        torch.cuda.synchronize()
        fa2_t = start.elapsed_time(end) / repeats
        
        print(f"K={test_K}: FA1={fa1_t:.3f}ms, FA2={fa2_t:.3f}ms, speedup={fa1_t/fa2_t:.2f}x")
