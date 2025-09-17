# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import contextmanager
from typing import Callable, Generator, Optional

import torch
import torch.nn.functional as F

# pip install git+https://github.com/mit-han-lab/Block-Sparse-Attention.git
from block_sparse_attn import block_sparse_attn_func
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import repeat_kv

from nncf.quantization.advanced_parameters import KVCachePrefillMode


class SparsePrefill:
    def __init__(
        self,
        algorithm: KVCachePrefillMode,
        threshold: float = 0.8,
        last_query_size: int = 100,
        recent_size: int = 1920,
        return_attn_scores: bool = False,
    ):
        self.algorithm = algorithm
        self.threshold = threshold
        self.last_query_size = last_query_size
        self.recent_size = recent_size
        self.return_attn_scores = return_attn_scores
        self._original_attn_impl = None

    def _get_prefill_impl(self) -> Callable:
        if self.algorithm == KVCachePrefillMode.XATTN:
            return xattention_forward
        elif self.algorithm == KVCachePrefillMode.TRI_SHAPE:
            return tri_shape_forward
        elif self.algorithm == KVCachePrefillMode.DENSE:
            return dense_forward
        else:
            error_msg = f"Unsupported KV cache prefill mode: {self.algorithm}"
            raise ValueError(error_msg)

    def store_original_forward(self, model: PreTrainedModel):
        if "Qwen2_5_VLForConditionalGeneration" in model.config.architectures:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward

            self._original_attn_impl = eager_attention_forward
        elif "Qwen2VLForConditionalGeneration" in model.config.architectures:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import eager_attention_forward

            self._original_attn_impl = eager_attention_forward
        elif "LlavaForConditionalGeneration" in model.config.architectures:
            from transformers.models.llama.modeling_llama import eager_attention_forward

            self._original_attn_impl = eager_attention_forward
        elif "LlavaNextForConditionalGeneration" in model.config.architectures:
            if "LlamaForCausalLM" in model.config.text_config.architectures:
                from transformers.models.llama.modeling_llama import eager_attention_forward
            elif "MistralForCausalLM" in model.config.text_config.architectures:
                from transformers.models.mistral.modeling_mistral import eager_attention_forward
            self._original_attn_impl = eager_attention_forward
        else:
            error_msg = f"Unsupported model class for: {model.config.architectures[0]}"
            raise ValueError(error_msg)

    def reset_forward(self, model: PreTrainedModel, eager_impl: Callable):
        if "Qwen2_5_VLForConditionalGeneration" in model.config.architectures:
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as modeling

            modeling.eager_attention_forward = eager_impl
        elif "Qwen2VLForConditionalGeneration" in model.config.architectures:
            import transformers.models.qwen2_vl.modeling_qwen2_vl as modeling

            modeling.eager_attention_forward = eager_impl
        elif "LlavaForConditionalGeneration" in model.config.architectures:
            import transformers.models.llama.modeling_llama as modeling

            modeling.eager_attention_forward = eager_impl
        elif "LlavaNextForConditionalGeneration" in model.config.architectures:
            if "LlamaForCausalLM" in model.config.text_config.architectures:
                import transformers.models.llama.modeling_llama as modeling
            elif "MistralForCausalLM" in model.config.text_config.architectures:
                import transformers.models.mistral.modeling_mistral as modeling
            modeling.eager_attention_forward = eager_impl
        else:
            error_msg = f"Unsupported model class for: {model.config.architectures[0]}"
            raise ValueError(error_msg)

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        try:
            llm = model
            if hasattr(llm, "model"):
                llm = llm.model
            if hasattr(llm, "language_model"):
                llm = llm.language_model

            assert llm.config._attn_implementation == "eager"

            self.store_original_forward(model)
            attention_interface = self._get_prefill_impl()
            self.reset_forward(model, attention_interface)

            yield
        finally:
            self.reset_forward(model, self._original_attn_impl)


def dense_forward(
    module: nn.Module,
    query: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    if key_states.shape[1] != query.shape[1]:
        key_states = repeat_kv(key_states, module.num_key_value_groups)
        value_states = repeat_kv(value_states, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights if kwargs.get("return_attn_scores", False) else None


def tri_shape_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    # Sparse Attention is not supported on decoding
    if query_states.shape[-2] == 1:
        return dense_forward(module, query_states, key_states, value_states, attention_mask, scaling)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    last_query_size = kwargs.get("last_query_size", 100)
    if kwargs.get("return_attn_scores", False):
        query_states_2 = query_states[:, :, -last_query_size:]
        attention_mask = attention_mask[:, :, -last_query_size:] if attention_mask is not None else None
        _, attn_weights = dense_forward(module, query_states_2, key_states, value_states, attention_mask, scaling)
    else:
        attn_weights = None

    batch_size, num_head, q_len, head_dim = query_states.shape
    k_len = key_states.shape[2]
    assert k_len == q_len
    block_size = 128
    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=query_states.device)
    k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=query_states.device)
    head_mask_type = torch.tensor([1 for _ in range(num_head)], device=query_states.device, dtype=torch.int32)
    simple_masks = torch.zeros(
        (1, num_head, q_block_num, k_block_num),
        dtype=torch.bool,
        device=query_states.device,
    )
    # keep_sink
    simple_masks[:, :, 0, :] = True  # keep first 128 tokens

    # keep_recent
    recent_size = kwargs.get("recent_size", 1024)
    local_block_num = (recent_size + block_size - 1) // block_size
    q_idx = torch.arange(q_block_num, device=simple_masks.device).unsqueeze(1)  # shape: [Q, 1]
    k_idx = torch.arange(k_block_num, device=simple_masks.device).unsqueeze(0)  # shape: [1, K]

    keep_recent_mask = (k_idx <= q_idx) & (k_idx >= (q_idx - local_block_num))  # [Q, K]
    keep_recent_mask = (
        keep_recent_mask.unsqueeze(0).unsqueeze(0).expand(1, num_head, q_block_num, k_block_num).to(simple_masks.device)
    )
    simple_masks |= keep_recent_mask

    # keep_q_lasts
    padded_len = q_block_num * block_size - q_len
    last_blocks_to_keep = (
        last_query_size + padded_len + block_size - 1
    ) // block_size  # ceil(a / b) == (a + b - 1) // b
    keep_q_lasts_mask = (k_idx <= q_idx) & (q_idx >= (q_block_num - last_blocks_to_keep))
    keep_q_lasts_mask = (
        keep_q_lasts_mask.unsqueeze(0)
        .unsqueeze(0)
        .expand(1, num_head, q_block_num, k_block_num)
        .to(simple_masks.device)
    )
    simple_masks |= keep_q_lasts_mask

    query_states = query_states.transpose(1, 2).view(q_len, num_head, head_dim)
    key_states = key_states.transpose(1, 2).view(k_len, num_head, head_dim).to(query_states.device)
    value_states = value_states.transpose(1, 2).view(k_len, num_head, head_dim).to(query_states.device)

    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        simple_masks[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=True,
    )
    attn_output = attn_output.view(batch_size, q_len, num_head, head_dim)

    return attn_output, attn_weights


def xattention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    stride: int = 16,
    block_size: int = 128,
    **kwargs,
):
    # Sparse Attention is not supported on decoding
    if query_states.shape[-2] == 1:
        return dense_forward(module, query_states, key_states, value_states, attention_mask, scaling)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    _, num_q_head, q_len, _ = query_states.shape
    assert num_q_head == num_kv_head

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size

    chunk_size = int(
        max(
            min(
                max(2048, 1 << (k_len - 1).bit_length()),  # next power of two ≥ k_len
                128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),  # # upper bound
            ),
            2048,  # lower bound
        )
    )

    last_query_size = kwargs.get("last_query_size", 100)
    if kwargs.get("return_attn_scores", False):
        query_states_2 = query_states[:, :, -last_query_size:]
        attention_mask = attention_mask[:, :, -last_query_size:] if attention_mask is not None else None
        _, attn_weights = dense_forward(module, query_states_2, key_states, value_states, attention_mask, scaling)
    else:
        attn_weights = None

    _, approx_simple_mask = xattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        threshold=kwargs.get("threshold", 0.8),
        chunk_size=chunk_size,
        keep_sink=False,
        keep_recent=False,
    )

    if approx_simple_mask.shape[1] != num_kv_head:
        approx_simple_mask = approx_simple_mask.expand(-1, num_kv_head, -1, -1)

    assert block_size == 128  # dense attn for last block (n_last = 100 <= block_size = 128)
    assert batch_size == 1
    query_states = query_states.transpose(1, 2).view(q_len, num_kv_head, head_dim)
    key_states = key_states.transpose(1, 2).view(k_len, num_kv_head, head_dim)
    value_states = value_states.transpose(1, 2).view(k_len, num_kv_head, head_dim)
    q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=query_states.device)
    k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=query_states.device)
    head_mask_type = torch.tensor([1 for _ in range(num_kv_head)], device=query_states.device, dtype=torch.int32)
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device

    approx_simple_mask = approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous()
    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=True,
        return_attn_probs=False,
    )
    attn_output = attn_output.view(batch_size, q_len, num_kv_head, head_dim)  # .transpose(1, 2)

    # sparsity_level = 1 - approx_simple_mask.sum() / approx_simple_mask.nelement()
    # print(f"approximated prefilling Computation: {approx_simple_mask.sum() / approx_simple_mask.nelement()}")
    # return attn_output, (attn_weights, sparsity_level)
    return attn_output, attn_weights


def find_blocks_chunked(input_tensor, current_index, threshold, decoding: bool, mode: str = "both", causal=True):
    """
    Finds and selects relevant blocks of attention for transformer-based models based on a
    threshold or a predefined number of blocks.

    Parameters
    ----------
    - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
    - current_index (int): The current index in the sequence processing.
    - threshold (float or None): A threshold value used to determine the minimum attention weight sum.
    - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
    - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
    - causal (bool): If True, applies causal masking to prevent future information leakage.

    Returns
    -------
    - torch.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
    indicating which blocks should be attended to.
    """
    assert threshold is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape

    if mode == "prefill" and decoding:
        return torch.ones_like(input_tensor, dtype=torch.bool)
    if mode == "decode" and not decoding:
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(1, head_num, chunk_num, chunk_num, device=input_tensor.device)
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(input_tensor, dtype=torch.bool)[:, :, 0 : current_index + 1],
                    torch.zeros_like(input_tensor, dtype=torch.bool)[:, :, current_index + 1 :],
                ],
                dim=-1,
            )
        else:
            return mask
    input_tensor = input_tensor.to(float)

    total_sum = input_tensor.sum(dim=-1, keepdim=True)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(float)
        required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            (batch_size, head_num, chunk_num, 1)
        ).to(input_tensor.device)
    else:
        required_sum = total_sum * threshold
    if causal:
        mask = torch.zeros_like(input_tensor, dtype=torch.bool)
        mask[:, :, :, 0] = 1  # keep the first key block
        mask[:, :, :, current_index : current_index + chunk_num] = (  # keep the diagonal block
            torch.eye(chunk_num, device=mask.device).unsqueeze(0).unsqueeze(0).expand(1, head_num, chunk_num, chunk_num)
        )
        other_values = input_tensor.masked_fill(mask, 0)
        sorted_values, _ = torch.sort(other_values, dim=-1, descending=True)
        sorted_values = sorted_values.to(input_tensor.device)

        sorted_values = torch.cat(
            [
                torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),  # cumulative logic
                torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),  # diagonal values
                sorted_values[:, :, :, :-2],
            ],
            dim=-1,
        )

        _, index = torch.sort(
            torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
            dim=-1,
            descending=True,
        )
        cumulative_sum_without_self = torch.cat(
            [
                torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)

        index_mask = cumulative_sum_without_self < required_sum
        index = torch.where(index_mask, index, 0)
        mask = mask.view(batch_size, head_num * chunk_num, block_num)
        index = index.view(batch_size, head_num * chunk_num, block_num)
        mask[:, torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1), index] = True
        mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        mask = torch.zeros_like(input_tensor, dtype=torch.bool)
        sorted_values, index = torch.sort(input_tensor, dim=-1, descending=True)
        sorted_values = sorted_values.to(input_tensor.device)
        cumulative_sum_without_self = torch.cat(
            [
                torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)
        index_mask = cumulative_sum_without_self < required_sum
        index = torch.where(index_mask, index, 0)
        mask = mask.view(batch_size, head_num * chunk_num, block_num)
        index = index.view(batch_size, head_num * chunk_num, block_num)
        mask[
            :,
            torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
            index,
        ] = True
        mask = mask.view(batch_size, head_num, chunk_num, block_num)

    try:
        if causal:
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except Exception:
        mask[:, :, :, current_index + chunk_num :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(input_tensor, dtype=bool, device=input_tensor.device)
            lambda_mask[:, :, :, 0] = 1
            lambda_mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=lambda_mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            assert torch.where(lambda_mask, mask, True).all()

    return mask


def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int = 128,
    stride: int = 16,
    threshold: float = 0.8,
    chunk_size: int = 2048,
    keep_sink: bool = False,
    keep_recent: bool = False,
    keep_q_lasts: bool = False,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0)
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0)
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1)
    reshaped_query = torch.cat(
        [(pad_query_states[:, :, (stride - 1 - q) :: stride, :]) for q in range(stride)],
        dim=-1,
    )
    assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        chunked_query = reshaped_query[
            :,
            :,
            (chunk_idx * reshaped_chunk_size) : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size),
            :,
        ]
        attn_weights_slice = torch.matmul(chunked_query, reshaped_key.transpose(2, 3))

        attn_weights_slice = attn_weights_slice / (head_dim**0.5) / stride

        causal_mask = torch.zeros(
            (
                batch_size,
                num_q_head,
                reshaped_chunk_size,
                reshaped_chunk_size * k_chunk_num,
            ),
            device=key_states.device,
        )
        causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
        chunk_start = chunk_idx * reshaped_chunk_size
        chunk_end = chunk_start + reshaped_chunk_size
        causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
            torch.ones(
                1,
                num_q_head,
                reshaped_chunk_size,
                reshaped_chunk_size,
                device=key_states.device,
            )
            * float("-inf"),
            diagonal=1,
        )

        if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
            causal_mask[:, :, (-(q_reshaped_num_to_pad)):, :] = float("-inf")

        causal_mask[:, :, :, chunk_end:] = float("-inf")
        attn_weights_slice = attn_weights_slice + causal_mask.to(attn_weights_slice.device)

        attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32).to(pad_query_states.dtype)

        if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
            attn_weights_slice[:, :, -q_reshaped_num_to_pad:, :] = 0

        attn_sum = (
            attn_weights_slice.view(
                batch_size,
                num_kv_head,
                num_blocks_per_chunk,
                reshaped_block_size,
                -1,
                reshaped_block_size,
            )
            .sum(dim=-1)
            .sum(dim=-2)
            .sum(dim=1, keepdim=True)  # GenAI aggregation accross heads
        )  # attention mass per block
        del chunked_query

        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=True,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
        torch.tril(
            torch.ones(q_block_num, q_block_num, dtype=bool, device=key_states.device),
            diagonal=0,
        ),
        simple_masks[:, :, -q_block_num:, -q_block_num:],
        False,
    )
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = eye_matrix.unsqueeze(0).unsqueeze(0).expand(1, num_kv_head, q_block_num, q_block_num)
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )
    if keep_q_lasts:
        q_last_tokens = 100
        q_blocks_to_keep = (q_last_tokens + block_size - 1) // block_size

        q_rows = torch.arange(q_block_num, device=simple_masks.device)
        q_keep_mask = q_rows >= (q_block_num - q_blocks_to_keep)

        q_keep_mask = q_keep_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, q_block_num, 1)
        q_keep_mask = q_keep_mask.expand(1, num_kv_head, q_block_num, k_block_num)
        simple_masks[:, :, -q_block_num:, :] = torch.where(q_keep_mask, True, simple_masks[:, :, -q_block_num:, :])

    return attn_sums, simple_masks
