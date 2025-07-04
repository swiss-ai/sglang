import ctypes
import os

import torch
import shutil
from pathlib import Path

def _find_cuda_home():
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            cuda_home = '/usr/local/cuda'
    return cuda_home

cuda_home = Path(_find_cuda_home())
if (cuda_home / 'lib').is_dir():
    cuda_path = cuda_home / 'lib'
elif (cuda_home / 'lib64').is_dir():
    cuda_path = cuda_home / 'lib64'
else:
    # Search for 'libcudart.so.12' in subdirectories
    for path in cuda_home.rglob('libcudart.so.12'):
        cuda_path = path.parent
        break
    else:
        raise RuntimeError("Could not find CUDA lib directory.")
cuda_include = (cuda_path / 'libcudart.so.12').resolve()
if cuda_include.exists():
    ctypes.CDLL(str(cuda_include), mode=ctypes.RTLD_GLOBAL)

from sgl_kernel import common_ops
from sgl_kernel.allreduce import *
from sgl_kernel.attention import (
    cutlass_mla_decode,
    cutlass_mla_get_workspace_size,
    lightning_attention_decode,
    merge_state,
    merge_state_v2,
)
from sgl_kernel.elementwise import (
    apply_rope_with_cos_sin_cache_inplace,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    silu_and_mul,
)
from sgl_kernel.gemm import (
    awq_dequantize,
    bmm_fp8,
    cutlass_scaled_fp4_mm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    int8_scaled_mm,
    qserve_w4a8_per_chn_gemm,
    qserve_w4a8_per_group_gemm,
    scaled_fp4_experts_quant,
    scaled_fp4_quant,
    sgl_per_tensor_quant_fp8,
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
    shuffle_rows,
)
from sgl_kernel.grammar import apply_token_bitmask_inplace_cuda
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    cutlass_fp4_group_mm,
    ep_moe_post_reorder,
    ep_moe_pre_reorder,
    fp8_blockwise_scaled_grouped_mm,
    moe_align_block_size,
    moe_fused_gate,
    prepare_moe_input,
    topk_softmax,
)
from sgl_kernel.sampling import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    top_p_sampling_from_probs,
)
from sgl_kernel.speculative import (
    build_tree_kernel_efficient,
    segment_packbits,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)
from sgl_kernel.version import __version__

build_tree_kernel = (
    None  # TODO(ying): remove this after updating the sglang python code.
)
