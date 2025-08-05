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
from nncf.quantization.advanced_parameters import KVCacheRefinedSelection


class KVCacheCompressor:
    def __init__(self, eviction_parameters: KVCacheCompressionParameters = KVCacheCompressionParameters()):
        self.algorithm = eviction_parameters.algorithm
        self.window_size = eviction_parameters.window_size
        if self.algorithm == KVCacheCompressionMode.SNAPKV and self.window_size is None:
            self.window_size = 8  # Default value for SNAPKV if not specified

        self.start_size = eviction_parameters.start_size
        self.recent_size = eviction_parameters.recent_size
        self.intermediate_size = eviction_parameters.intermediate_size
        self.refined_size = eviction_parameters.refined_size
        self.refined_algorithm = eviction_parameters.refined_algorithm
        self.score_aggregation = eviction_parameters.score_aggregation
        self.strategy = eviction_parameters.strategy
        self.group_size = eviction_parameters.group_size if self.strategy == "per_group" else 1
        self.apply_rerotation = eviction_parameters.apply_rerotation
        self.kvcrush_anchor = eviction_parameters.kvcrush_anchor
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
        if self.start_size < 0 or self.recent_size < 0 or self.intermediate_size < 0 or self.refined_size < 0:
            error_msg = "KV cache sizes must be non-negative integers."
        elif self.refined_size > self.intermediate_size:
            error_msg = "refined_size cannot be greater than intermediate_size."
        elif self.start_size + self.recent_size + self.intermediate_size + self.refined_size <= 0:
            error_msg = "At least one of the KV cache sizes must be greater than zero."
        elif any(
            size % self.group_size != 0
            for size in (self.start_size, self.recent_size, self.intermediate_size, self.refined_size)
        ):
            error_msg = "KV cache part sizes must be divisible by the group size."
        elif self.score_aggregation not in {"sum", "norm_sum"}:
            error_msg = "score_aggregation must be either 'sum' or 'norm_sum'."
        elif self.window_size is not None and self.algorithm != KVCacheCompressionMode.SNAPKV:
            error_msg = "Window size is only supported for SNAPKV algorithm."
        elif self.window_size is not None and self.window_size <= 0:
            error_msg = "Window size must be a positive integer if specified."
        elif self.strategy not in {"per_token", "per_group"}:
            error_msg = f"Strategy {self.strategy} is not supported. Supported strategies: 'per_token'."
        elif self.kvcrush_anchor not in {"random", "zeros", "ones", "mean", "alternate"}:
            error_msg = (
                f"Unknown KVCrush anchor: {self.kvcrush_anchor}. "
                "Supported anchors: 'random', 'zeros', 'ones', 'mean', 'alternate'."
            )

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
            hh_score = attn_w.sum(0).sum(1)  # Sum over batch and query length, shape: (H, seq_len)

        # num_attn_heads = hh_score.shape[0]
        # if num_attn_heads != 1:
        hh_score = hh_score.sum(0, keepdim=True)  # Sum over all heads, shape: (1, seq_len)

        # Skip frozen start tokens in cache
        # TODO: What if prompt size is smaller than start_size? - Remove slicing
        hh_score = hh_score[:, self.start_size :]

        if layer_scores is None:
            layer_scores = hh_score
            # if layer_idx == 0:
            #     print("hh_score.shape", hh_score.shape, "attn_w", attn_w.shape)
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
            # if layer_idx == 0:
            #     print("hh_score.shape", hh_score.shape, "layer_scores.shape", layer_scores.shape, "attn_w", attn_w.shape)
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
        group_scores = group_scores.sum(-1) if self.score_aggregation == "sum" else group_scores.max(-1).values

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

    def get_refined_indices(self, scores: torch.Tensor, kwargs: dict) -> torch.Tensor:
        if self.refined_algorithm == KVCacheRefinedSelection.KVCRUSH:
            B, _ = scores.shape
            if B != 1:
                error_msg = "KVCacheCompressor with KVCrush algorithm supports only batch size of 1."
                raise ValueError(error_msg)

            scores_flat = scores.view(-1)
            refined_mask = scores_flat != float("-inf")
            keepable_scores = scores_flat[refined_mask]

            # Binary vector: top 50% → 1, bottom 50% → 0
            num_zeros = keepable_scores.numel() // 2
            _, low_idx = torch.topk(keepable_scores, num_zeros, largest=False)
            binary_vector = torch.ones_like(keepable_scores, dtype=torch.int)
            binary_vector[low_idx] = 0

            # Place binary_vector back into full-length binary tensor
            full_binary = torch.zeros_like(scores_flat, dtype=torch.int, device=scores.device)
            full_binary[refined_mask] = binary_vector

            if self.strategy == "per_group":
                full_binary = full_binary.view(-1, self.group_size)
                num_groups = full_binary.shape[0]

                if self.kvcrush_anchor == "random":
                    anchor_point = torch.randint(0, 2, (num_groups,), device=scores.device)
                elif self.kvcrush_anchor == "zeros":
                    anchor_point = torch.zeros(num_groups)
                elif self.kvcrush_anchor == "ones":
                    anchor_point = torch.ones(num_groups)
                elif self.kvcrush_anchor == "mean":
                    mean_point = full_binary.float().mean(dim=1)
                    anchor_point = (mean_point > 0.5).int()
                elif self.kvcrush_anchor == "alternate":
                    anchor_point = torch.zeros(num_groups)
                    anchor_point[1::2] = 1

                hamming_distance = torch.sum(
                    full_binary != anchor_point.unsqueeze(1), dim=1
                ).float()  # shape: [num_groups]
                refined_group_mask = refined_mask.view(-1, self.group_size)[:, 0]
                hamming_distance[~refined_group_mask] = float("-inf")  # Set invalid indices to -inf

                sorted_dist_idx = torch.argsort(hamming_distance, descending=True)

                # Select evenly spaced indices using linspace (representative)
                num_valid = keepable_scores.numel() // self.group_size
                rep_indices = torch.linspace(
                    0, num_valid - 1, steps=self.refined_size // self.group_size, dtype=torch.long, device=scores.device
                )
                assert rep_indices.numel() == self.refined_size // self.group_size
                refined_topk = sorted_dist_idx[rep_indices]  # shape: [refined_groups]

                return refined_topk

            # Anchor: shape [L]
            if self.kvcrush_anchor == "random":
                anchor = torch.randint_like(keepable_scores, low=0, high=2, device=scores.device)
            elif self.kvcrush_anchor == "zeros":
                anchor = torch.zeros_like(keepable_scores, dtype=torch.int, device=scores.device)
            elif self.kvcrush_anchor == "ones":
                anchor = torch.ones_like(keepable_scores, dtype=torch.int, device=scores.device)
            elif self.kvcrush_anchor == "mean":  # equal to binary_vector in per-token case
                error_msg = (
                    "Mean anchor is not supported for KVCrush in per-token mode. "
                    "Please use 'random', 'zeros', 'ones' or 'alternate' anchors."
                )
                raise ValueError(error_msg)
            elif self.kvcrush_anchor == "alternate":
                anchor = torch.zeros_like(keepable_scores, dtype=torch.int, device=scores.device)
                anchor[1::2] = 1

            full_anchor = torch.zeros_like(scores_flat, dtype=torch.int)
            full_anchor[refined_mask] = anchor

            # Hamming distance (1D): count bits different from anchor
            hamming_distance = (full_binary != full_anchor).float()
            hamming_distance[~refined_mask] = float("-inf")  # Set invalid indices to -inf

            # Sort valid indices by distance to anchor (more diverse first)
            sorted_dist_idx = torch.argsort(hamming_distance, descending=True)

            # Select evenly spaced indices using linspace (representative)
            num_valid = keepable_scores.numel()
            rep_indices = torch.linspace(
                0, num_valid - 1, steps=self.refined_size, dtype=torch.long, device=scores.device
            )
            assert rep_indices.numel() == self.refined_size
            refined_topk = sorted_dist_idx[rep_indices].unsqueeze(0)  # shape: [1, refined_size]

        elif self.refined_algorithm == KVCacheRefinedSelection.CRITICALKV:
            # Minimize the output perturbation - how much the model's output changes
            # when certain KV entries are removed. L1 distance is used to measure the perturbation.
            values = kwargs.get("values")
            W_O = kwargs.get("W_O")
            eps = 1e-4

            V_proj = values @ W_O
            # Compute L1 norm of each projected value vector
            value_norms = V_proj.norm(p=1, dim=1)
            # select only intermediate part
            value_norms = value_norms[self.start_size : self.start_size + scores.shape[-1]]

            # Adjust scores to reflect both attention and value importance
            adjusted_scores = (scores + eps) * value_norms

            if self.strategy == "per_group":
                adjusted_scores = adjusted_scores.view(-1, self.group_size).sum(dim=-1)  # Sum token scores inside group

            refined_size = self.refined_size // self.group_size
            _, refined_topk = torch.topk(adjusted_scores, refined_size, dim=-1)
        return refined_topk

    def _get_per_token_indices(self, scores: torch.Tensor, kwargs: dict) -> torch.Tensor:
        keep = []
        if self.start_size > 0:
            keep_past = torch.arange(0, self.start_size, device=scores.device).unsqueeze(0)
            keep.append(keep_past)

        if self.intermediate_size > 0:
            intermediate_scores = scores[:, : scores.shape[-1] - self.recent_size]

            # Split into coarse (for primary algorithm) and refined parts (for secondary algorithm)
            coarse_size = self.intermediate_size - self.refined_size
            if coarse_size > 0:
                _, coarse_topk = torch.topk(intermediate_scores, coarse_size, dim=-1)
                coarse_topk = coarse_topk.sort().values + self.start_size
                keep.append(coarse_topk)

            if self.refined_size > 0:
                # Mask coarse indices before refined selection
                coarse_idx = coarse_topk - self.start_size  # [1, coarse_size]
                mask = torch.zeros_like(intermediate_scores, dtype=torch.bool)
                mask.scatter_(1, coarse_idx, True)
                masked_scores = intermediate_scores.masked_fill(mask, float("-inf"))

                refined_topk = self.get_refined_indices(masked_scores, kwargs) + self.start_size
                keep.append(refined_topk)

        if self.recent_size > 0:
            seq_len = self.start_size + scores.shape[-1]
            keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=scores.device).unsqueeze(0)
            keep.append(keep_recent)

        remaining_idx = torch.cat(keep, dim=-1)
        return remaining_idx

    def _get_per_group_indices(self, scores: torch.Tensor, kwargs: dict) -> torch.Tensor:
        # Pad scores with zeros to make it multiple by group_size
        seq_len = self.start_size + scores.shape[-1]
        pad = scores.shape[-1] % self.group_size
        if pad:
            scores = F.pad(scores, (0, self.group_size - pad), mode="constant", value=0)
        group_scores = scores.view(-1, self.group_size).sum(-1)  # Sum token scores inside group

        keep_groups = []
        num_start_groups = self.start_size // self.group_size
        num_recent_groups = self.recent_size // self.group_size
        if self.start_size > 0:
            keep_past = torch.arange(0, num_start_groups, device=scores.device)
            keep_groups.append(keep_past)

        if self.intermediate_size > 0:
            inter_group_scores = group_scores[: group_scores.shape[-1] - num_recent_groups]
            num_coarse_groups = (self.intermediate_size - self.refined_size) // self.group_size
            if num_coarse_groups > 0:
                _, keep_coarse = torch.topk(inter_group_scores, num_coarse_groups, dim=-1)
                keep_coarse = keep_coarse.sort().values + num_start_groups
                keep_groups.append(keep_coarse)

            num_refined_groups = self.refined_size // self.group_size
            if num_refined_groups > 0:
                # Mask coarse indices before refined selection
                coarse_group_idx = keep_coarse - num_start_groups
                mask = torch.zeros_like(scores, dtype=torch.bool)
                coarse_idx = self._convert_group_indices(
                    coarse_group_idx.unsqueeze(0), coarse_group_idx.shape[0] * self.group_size
                )
                mask.scatter_(1, coarse_idx, True)
                masked_inter_scores = scores.masked_fill(mask, float("-inf"))[
                    :, : inter_group_scores.shape[0] * self.group_size
                ]

                refined_topk = self.get_refined_indices(masked_inter_scores, kwargs) + num_start_groups
                keep_groups.append(refined_topk)

        if self.recent_size > 0:
            num_groups = group_scores.shape[0]
            keep_recent = (
                torch.arange(num_groups - num_recent_groups, num_groups, device=scores.device) + num_start_groups
            )
            keep_groups.append(keep_recent)

        remaining_group_idx = torch.cat(keep_groups, dim=-1).unsqueeze(0)
        remaining_idx = self._convert_group_indices(remaining_group_idx, seq_len)
        return remaining_idx

    def get_remaining_indices(self, scores: torch.Tensor, kwargs: dict) -> torch.Tensor:
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
            remaining_idx = self._get_per_token_indices(scores, kwargs)
        elif self.strategy == "per_group":
            remaining_idx = self._get_per_group_indices(scores, kwargs)

        return remaining_idx

    def _get_rerotated_keys(self, key_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 temporarily for better accuracy
        B, H, seq_len_after_eviction, head_dim = key_states.shape
        dtype = key_states.dtype
        key_states = key_states.to(torch.float32)

        pos_ids = torch.arange(seq_len_after_eviction).unsqueeze(0).to(key_states.device)
        after_cos, after_sin = self.rotary_emb(key_states, pos_ids)

        cur_ids = indices.to(key_states.device)
        before_cos, before_sin = self.rotary_emb(key_states, cur_ids)

        rerotation_cos = after_cos * before_cos + after_sin * before_sin
        rerotation_sin = after_sin * before_cos - after_cos * before_sin

        rotated_key_states = key_states * rerotation_cos.unsqueeze(1) + rotate_half(key_states) * rerotation_sin.unsqueeze(1)
        return rotated_key_states.to(dtype)

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
        indices = self.get_remaining_indices(scores, kwargs)

        # Prune keys and values
        keep_heads = indices.shape[0]
        B, H, seq_len, head_dim = keys.shape
        mask = torch.zeros((keep_heads, seq_len), dtype=torch.bool).to(keys.device)  # shape (keep_heads, seq_len)
        mask = mask.scatter(-1, indices, 1)
        mask = mask.unsqueeze(0).unsqueeze(-1)

        keys = keys.masked_select(mask).view(B, H, -1, head_dim)
        values = values.masked_select(mask).view_as(keys)

        score_mask = mask[0, :, self.start_size :, 0]  # shape (keep_heads, seq_len - self.start_size,)
        self._scores[layer_idx] = self._scores[layer_idx].masked_select(score_mask).view(keep_heads, -1)

        if self.score_aggregation == "norm_sum":
            self._cache_counter[layer_idx] = self._cache_counter[layer_idx].masked_select(score_mask[0])

        # Apply keys rerotation
        if self.apply_rerotation:
            keys = self._get_rerotated_keys(keys, indices)

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
        if attn_weights is None:
            error_msg = "Attention weights are None. Please switch to the `eager` attention implementation"
            raise RuntimeError(error_msg)

        # TODO: Support chunked prefill (prev_attn_weights.shape[-2] == 1 and current_attn_weights.shape[-2] != 1)
        if layer_idx == 0 and attn_weights.shape[-2] != 1:
            self.clean()

        self._update_scores(layer_idx, attn_weights)

        if seq_len > self.max_cache_size:
            if self.refined_size > 0 and self.refined_algorithm == KVCacheRefinedSelection.CRITICALKV:
                kwargs["W_O"] = module.o_proj.weight
                kwargs["values"] = values.permute(0, 2, 1, 3).reshape(seq_len, -1)
            keys, values = self.compress(layer_idx, keys, values, kwargs)

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
            llm = model
            if hasattr(llm, "model"):
                llm = llm.model
            if hasattr(llm, "language_model"):
                llm = llm.language_model

            if self.apply_rerotation:
                if hasattr(llm, "rotary_emb"):
                    self.rotary_emb = llm.rotary_emb
                elif hasattr(llm, "layers"):
                    self.rotary_emb = llm.layers[0].self_attn.rotary_emb

            attn_layer = llm.layers[0].self_attn
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
            for layer in llm.layers:
                if getattr(layer.self_attn, "is_sliding", False):
                    nncf_logger.warning("Compression is skipped for layers with sliding window attention")
                    continue
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
