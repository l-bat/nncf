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
from typing import Generator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import rotate_half

from nncf import nncf_logger
from nncf.quantization.advanced_parameters import KVCacheCompressionMode
from nncf.quantization.advanced_parameters import KVCacheCompressionParameters


class KVCacheCompressor:
    def __init__(self, eviction_parameters: KVCacheCompressionParameters = KVCacheCompressionParameters()):
        self.algorithm = eviction_parameters.algorithm
        self.window_size = eviction_parameters.window_size
        if self.algorithm == KVCacheCompressionMode.SNAPKV and self.window_size is None:
            self.window_size = 8  # Default value for SNAPKV if not specified

        self.start_size = eviction_parameters.start_size
        self.recent_size = eviction_parameters.recent_size
        self.intermediate_size = eviction_parameters.intermediate_size
        self.score_aggregation = eviction_parameters.score_aggregation
        self.strategy = eviction_parameters.strategy
        self.group_size = eviction_parameters.group_size if self.strategy == "per_group" else 1
        self.apply_rerotation = eviction_parameters.apply_rerotation
        self._validate_arguments()

        self._scores = []
        self._cache_counter = None if self.score_aggregation == "sum" else []

        self._cos_cache = None
        self._sin_cache = None

    def _validate_arguments(self):
        """
        Validates the arguments for the KV Cache compressor.
        Raises a ValueError at the end if any condition fails.
        """
        error_msg = None
        if self.start_size <= 0 or self.recent_size <= 0 or self.intermediate_size <= 0:
            error_msg = "KV cache sizes must be positive integers."
        elif any(size % self.group_size != 0 for size in (self.start_size, self.recent_size, self.intermediate_size)):
            error_msg = "KV cache part sizes must be divisible by the group size."
        elif self.score_aggregation not in {"sum", "norm_sum"}:
            error_msg = "score_aggregation must be either 'sum' or 'norm_sum'."
        elif self.window_size is not None and self.algorithm != KVCacheCompressionMode.SNAPKV:
            error_msg = "Window size is only supported for SNAPKV algorithm."
        elif self.window_size is not None and self.window_size <= 0:
            error_msg = "Window size must be a positive integer if specified."
        elif self.strategy not in {"per_token", "per_group"}:
            error_msg = f"Strategy {self.strategy} is not supported. Supported strategies: 'per_token'."

        if error_msg:
            raise ValueError(error_msg)

    @property
    def max_cache_size(self) -> int:
        """
        Returns the maximum size of the KV cache.
        """
        return self.start_size + self.recent_size + self.intermediate_size

    def clean(self):
        """
        Resets the scores and cache counter.
        """
        self._scores = []
        self._cache_counter = None if self.score_aggregation == "sum" else []

        self._cos_cache = None
        self._sin_cache = None

    def _update_scores(self, layer_idx, attn_w):
        """
        Updates the scores based on the attention weights.
        """
        is_norm_sum = self.score_aggregation == "norm_sum"
        layer_scores = self._scores[layer_idx] if len(self._scores) > layer_idx else None
        layer_counter = self._cache_counter[layer_idx] if is_norm_sum and len(self._cache_counter) > layer_idx else None

        if self.window_size is not None:
            hh_score = attn_w[..., -self.window_size :, :].sum(0).sum(1)  # sum over batch and query length
            if layer_scores is None:
                hh_score = torch.max_pool1d(hh_score, kernel_size=7, padding=7 // 2, stride=1)
        else:
            hh_score = attn_w.sum(0).sum(1)

        # num_attn_heads = hh_score.shape[0]
        # if num_attn_heads != 1:
        hh_score = hh_score.sum(0, keepdim=True)  # Sum over all heads, shape: (1, seq_len)

        # Skip frozen start tokens in cache
        # TODO: What if prompt size is smaller than start_size? - Remove slicing
        hh_score = hh_score[:, self.start_size :]

        if layer_scores is None:
            layer_scores = hh_score
            if is_norm_sum:
                seq_len = layer_scores.shape[-1]  # seq_len without start_size
                if self.window_size is not None:
                    if seq_len > self.window_size:
                        full_window = torch.full(
                            (seq_len - self.window_size,),
                            self.window_size,
                            dtype=layer_scores.dtype,
                            device=layer_scores.device,
                        )
                        tail = torch.arange(
                            self.window_size, 0, -1, dtype=layer_scores.dtype, device=layer_scores.device
                        )
                        layer_counter = torch.cat((full_window, tail), dim=0)
                    else:
                        layer_counter = torch.arange(
                            seq_len, 0, -1, dtype=layer_scores.dtype, device=layer_scores.device
                        )
                else:
                    layer_counter = torch.arange(seq_len, 0, -1, dtype=layer_scores.dtype, device=layer_scores.device)
        else:
            num_new_tokens = hh_score.shape[-1] - layer_scores.shape[-1]
            hh_score[:, :-num_new_tokens] += layer_scores
            layer_scores = hh_score

            if is_norm_sum:
                if self.window_size is not None:
                    w_size = min(self.window_size, num_new_tokens)
                    new_tail = torch.arange(w_size, 0, -1, dtype=layer_scores.dtype, device=layer_scores.device)
                    layer_counter += w_size
                    layer_counter = torch.cat((layer_counter, new_tail), dim=-1)
                else:
                    layer_counter += num_new_tokens
                    new_counters = torch.arange(
                        num_new_tokens, 0, -1, dtype=layer_scores.dtype, device=layer_scores.device
                    )
                    layer_counter = torch.cat((layer_counter, new_counters), dim=-1)

        if len(self._scores) <= layer_idx:
            self._scores.append(layer_scores)
            if is_norm_sum:
                self._cache_counter.append(layer_counter)
        else:
            self._scores[layer_idx] = layer_scores
            if is_norm_sum:
                self._cache_counter[layer_idx] = layer_counter

    def get_scores(self, layer_idx):
        if self.score_aggregation == "sum":
            return self._scores[layer_idx]
        else:
            return self._scores[layer_idx] / self._cache_counter[layer_idx]

    def get_intermediate_page_scores(self):
        scores = self.get_scores()

        # Pad cache with zeros to make it multiple by group_size
        pad = scores.shape[-1] % self.group_size
        if pad:
            scores = F.pad(scores, (0, self.group_size - pad), mode="constant", value=0)

        group_scores = scores.view(self.num_heads_to_keep, -1, self.group_size)
        # TODO: Add norm group mode (divide by number of tokens in group if we use padding)
        group_scores = group_scores.sum(-1) if self.group_mode == "sum" else group_scores.max(-1).values

        num_recent_groups = self.recent_size // self.group_size
        intermediate_group_scores = group_scores[:, :-num_recent_groups]

        return intermediate_group_scores

    def _convert_group_indices(self, group_indices, seq_len):
        heads, num_groups = group_indices.shape
        device = group_indices.device

        # Create relative indices within each group
        relative_idx = torch.arange(self.group_size, device=device).repeat(num_groups)
        relative_idx = relative_idx.view(1, num_groups, self.group_size)

        expanded_groups = group_indices.unsqueeze(-1).expand(-1, -1, self.group_size)
        indices = expanded_groups * self.group_size + relative_idx
        indices = indices.view(heads, -1)

        # Trim padding from the last group if needed
        remainder = seq_len % self.group_size
        if remainder:
            padded = self.group_size - remainder
            indices = indices[:, :-padded]

        return indices

    def get_remaining_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Computes the indices of the keep tokens in the KV cache after compression.

        Parameters
        ----------
        scores : torch.Tensor
            Scores of the tokens in the intermediate and recent parts of KV cache
        Returns:
            torch.Tensor: Indices of the remaining tokens in the KV cache
        """
        if self.strategy == "per_token":
            keep_past = torch.arange(0, self.start_size, device=scores.device).unsqueeze(0)

            _, keep_topk = torch.topk(scores[:, : -self.recent_size], self.intermediate_size, dim=-1)
            keep_topk = keep_topk.sort().values + self.start_size
            seq_len = self.start_size + scores.shape[-1]
            keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=scores.device).unsqueeze(0)
            remaining_idx = torch.cat([keep_past, keep_topk, keep_recent], dim=-1)
        elif self.strategy == "per_group":
            # Pad scores with zeros to make it multiple by group_size
            seq_len = self.start_size + scores.shape[-1]
            pad = scores.shape[-1] % self.group_size
            if pad:
                scores = F.pad(scores, (0, self.group_size - pad), mode="constant", value=0)
            group_scores = scores.view(-1, self.group_size).sum(-1)  # Sum token scores inside group

            num_start_groups = self.start_size // self.group_size
            num_recent_groups = self.recent_size // self.group_size
            num_intermediate_groups = self.intermediate_size // self.group_size
            num_groups = group_scores.shape[0]

            _, keep_topk = torch.topk(group_scores[:-num_recent_groups], num_intermediate_groups, dim=-1)
            keep_topk = keep_topk.sort().values + num_start_groups

            keep_past = torch.arange(0, num_start_groups, device=scores.device)
            keep_recent = (
                torch.arange(num_groups - num_recent_groups, num_groups, device=scores.device) + num_start_groups
            )
            remaining_group_idx = torch.cat([keep_past, keep_topk, keep_recent], dim=-1).unsqueeze(0)
            remaining_idx = self._convert_group_indices(remaining_group_idx, seq_len)

        return remaining_idx

    def _get_rerotated_keys(self, key_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 temporarily for better accuracy
        seq_len_after_eviction = key_states.shape[-2]

        expected_cos = self._cos_cache[:, :seq_len_after_eviction, :].to(torch.float32)
        expected_sin = self._sin_cache[:, :seq_len_after_eviction, :].to(torch.float32)

        b, _, head_dim = self._cos_cache.shape
        rot_mask = mask[:, 0]  # shape (batch, seq_len_after_eviction, head_dim)
        current_cos = self._cos_cache.masked_select(rot_mask).view(b, -1, head_dim).to(torch.float32)
        current_sin = self._sin_cache.masked_select(rot_mask).view(b, -1, head_dim).to(torch.float32)

        rerotation_cos = expected_cos * current_cos + expected_sin * current_sin
        rerotation_sin = expected_sin * current_cos - expected_cos * current_sin

        rotated_key_states = key_states * rerotation_cos + rotate_half(key_states) * rerotation_sin
        return rotated_key_states

    def _update_cos_sin_cache(self, seq_len: int, kwargs: dict):
        """
        Updates the cos/sin caches if rerotation is applied.
        This is necessary to ensure that the keys are rerotated correctly after compression.
        """
        if self.apply_rerotation:
            cos, sin = kwargs["position_embeddings"]
            using_rope = cos is not None and sin is not None
            self.apply_rerotation = using_rope

            if using_rope:
                # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
                # after all RoPE models have a llama-like cache utilization.
                if cos.dim() == 2:
                    self._cos_cache = cos.unsqueeze(0)
                    self._sin_cache = sin.unsqueeze(0)
                elif self._cos_cache is None or cos.shape[1] > 1:
                    self._cos_cache = cos
                    self._sin_cache = sin
                elif seq_len < self._cos_cache.shape[1]:
                    self._cos_cache = self._cos_cache[:, :seq_len, :]
                    self._sin_cache = self._sin_cache[:, :seq_len, :]
                else:
                    self._cos_cache = torch.cat([self._cos_cache, cos], dim=1)
                    self._sin_cache = torch.cat([self._sin_cache, sin], dim=1)

    def compress(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method.

        Parameters
        ----------
        module :
            Transformer layer, see `hook` method for more details
        hidden_states :
            Hidden states of the layer
        keys :
            Keys of the cache
        values :
            Values of the cache
        attentions :
            Attention weights of the layer
        kwargs :
            Keyword arguments, as given to the forward pass of the layer

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated keys and values
        """
        # Compute scores
        scores = self.get_scores(layer_idx)
        assert scores.shape[-1] > self.recent_size
        indices = self.get_remaining_indices(scores)

        # Prune keys and values
        mask = torch.zeros((indices.shape[0], keys.shape[-2]), dtype=torch.bool).to(keys.device)
        mask = mask.scatter(-1, indices, 1)
        mask = mask.unsqueeze(0).unsqueeze(-1)  # Add batch and head_dim dims, shape (B, H, upd_seq_len, head_dim)

        keys = keys.masked_select(mask).view(keys.shape[0], keys.shape[1], -1, keys.shape[-1])
        values = values.masked_select(mask).view(values.shape[0], values.shape[1], -1, values.shape[-1])

        score_mask = mask[0, :, self.start_size :, 0]  # shape (upd_seq_len,)
        self._scores[layer_idx] = (
            self._scores[layer_idx].masked_select(score_mask).view(self._scores[layer_idx].shape[0], -1)
        )
        if self.score_aggregation == "norm_sum":
            self._cache_counter[layer_idx] = self._cache_counter[layer_idx].masked_select(score_mask[0])

        # Apply keys rerotation
        if self.apply_rerotation:
            keys = self._get_rerotated_keys(keys, mask)

        return keys, values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.
        The hook calls the compress method to compress the KV cache.

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.
        """
        layer_idx = module.layer_idx
        cache = kwargs["past_key_value"]
        keys = cache.key_cache[layer_idx]
        values = cache.value_cache[layer_idx]
        seq_len = keys.shape[-2]
        attn_weights = output[1]

        # TODO: Support chunked prefill (prev_attn_weights.shape[-2] == 1 and current_attn_weights.shape[-2] != 1)
        if layer_idx == 0 and attn_weights.shape[-2] != 1:
            self.clean()

        self._update_scores(layer_idx, attn_weights)

        # Update cos/sin caches if rerotation is applied
        if layer_idx == 0 and self.apply_rerotation:
            self._update_cos_sin_cache(seq_len, kwargs)

        if layer_idx == 0:
            print(f"Current sequence length: {seq_len}, max cache size: {self.max_cache_size}")
        if seq_len > self.max_cache_size:
            if layer_idx == 0:
                print("Before compression:", keys.shape)
            keys, values = self.compress(layer_idx, keys, values, kwargs)
            if layer_idx == 0:
                print("After compression:", keys.shape)

        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.

        Parameters
        ----------
        model : PreTrainedModel
            Model to apply the compression method to
        """
        hooks = []
        try:
            attn_layer = model.model.layers[0].self_attn
            if (
                self.apply_rerotation
                and getattr(attn_layer, "rotary_ndims", None) is not None
                or getattr(attn_layer, "rotary_dim", None) is not None
            ):
                self.apply_rerotation = False
                nncf_logger.warning(
                    "Rerotation is not supported for models with partially rotated position embeddings. "
                    "Compression will be applied without rerotation."
                )
            for layer in model.model.layers:
                if getattr(layer.self_attn, "is_sliding", False):
                    nncf_logger.warning("Compression is skipped for layers with sliding window attention")
                    continue
                layer.self_attn.rotary_emb = model.model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
