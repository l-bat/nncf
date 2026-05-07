# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Iterator, Optional, Union

import mlflow
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset

# from lm_eval import simple_evaluate
# from lm_eval.models.optimum_lm import OptimumLM
# from optimum.exporters.openvino.convert import export_from_model
# from optimum.intel.openvino import OVModelForCausalLM
# from optimum.modeling_base import OptimizedModel
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedAWQParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import StretchedSymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.quantize_functions import set_use_autograd_quantize

warnings.filterwarnings("ignore", category=TracerWarning)


def _log_dataset_size(name: str, found: int, requested: int) -> None:
    if found < requested:
        print(f"[dataset] {name}: {found} unique samples pass the seqlen filter (requested {requested})")
    else:
        print(f"[dataset] {name}: {found} samples will be used for tuning")


def get_wikitext2(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param num_samples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    limit = num_samples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(num_samples):
        # Crop a sequence of tokens of length seqlen starting at a random position
        i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
    return trainloader


def get_pile(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device):
    # def preprocess_fn(example):
    #     return {"text": tokenizer.apply_chat_template(example["text"], add_generation_prompt=False, tokenize=False)}
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    # ds = ds.shuffle(seed=42).select(range(10 * num_samples))
    # ds = ds.map(preprocess_fn)

    trainloader = []
    for example in ds:
        trainenc = tokenizer(example["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break

    _log_dataset_size("pile", len(trainloader), num_samples)
    return trainloader


def get_slimpajama(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device):
    ds = load_dataset("DKYoon/SlimPajama-6B", split="train")
    trainloader = []
    for example in ds:
        trainenc = tokenizer(example["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break
    _log_dataset_size("slimpajama", len(trainloader), num_samples)
    return trainloader


def get_metamathqa(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    trainloader = []
    for example in ds:
        text = example["query"] + "\n" + example["response"]
        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break
    _log_dataset_size("metamathqa", len(trainloader), num_samples)
    return trainloader


def get_numina_math(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device):
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    trainloader = []
    for example in ds:
        text = example["problem"] + "\n" + example["solution"]
        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        inp = trainenc.input_ids[:, i : i + seqlen].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break
    _log_dataset_size("numina_math", len(trainloader), num_samples)
    return trainloader


DATASET_LOADERS = {
    "pile": get_pile,
    "wikitext": get_wikitext2,
    "slimpajama": get_slimpajama,
    "metamathqa": get_metamathqa,
    "numina_math": get_numina_math,
}


# ── Streaming data & sharded hiddens (for LR-QAT-style unique-data-per-step training) ──


def iter_tokenized_samples(
    dataset_name: str,
    seqlen: int,
    tokenizer: Any,
    device: torch.device,
    seed: int = 42,
) -> Iterator[Tensor]:
    """
    Infinite iterator that yields tokenized samples of length ``seqlen`` from a HF dataset.

    Streams through the dataset once, then reshuffles and repeats (with a different seed each pass).
    Each sample is a ``[1, seqlen]`` tensor of input_ids on ``device``.
    """
    pass_num = 0
    while True:
        if dataset_name == "slimpajama":
            ds = load_dataset("DKYoon/SlimPajama-6B", split="train")
        elif dataset_name == "pile":
            ds = load_dataset("NeelNanda/pile-10k", split="train")
        elif dataset_name == "metamathqa":
            ds = load_dataset("meta-math/MetaMathQA", split="train")
        elif dataset_name == "numina_math":
            ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        else:
            error = f"Streaming not supported for dataset '{dataset_name}'."
            raise ValueError(error)
        ds = ds.shuffle(seed=seed + pass_num)
        for example in ds:
            if dataset_name == "metamathqa":
                text = example["query"] + "\n" + example["response"]
            elif dataset_name == "numina_math":
                text = example["problem"] + "\n" + example["solution"]
            else:
                text = example["text"]
            enc = tokenizer(text, return_tensors="pt")
            if enc.input_ids.shape[1] < seqlen:
                continue
            if enc.input_ids.shape[1] > seqlen + 1:
                i = torch.randint(0, enc.input_ids.shape[1] - seqlen - 1, (1,)).item()
            else:
                i = 0
            yield enc.input_ids[:, i : i + seqlen].to(device)
        pass_num += 1


def collect_samples(
    dataset_name: str, num_samples: int, seqlen: int, tokenizer: Any, device: torch.device
) -> list[Tensor]:
    """Collect a fixed number of samples from the streaming iterator."""
    it = iter_tokenized_samples(dataset_name, seqlen, tokenizer, device)
    samples = [next(it) for _ in range(num_samples)]
    _log_dataset_size(dataset_name, len(samples), num_samples)
    return samples


@torch.no_grad()
def calc_hiddens_sharded(
    model: nn.Module,
    samples: list[Tensor],
    hiddens_dir: Path,
    shard_size: int = 512,
) -> int:
    """
    Compute teacher hidden states and save them to disk in shards.

    Each shard file contains a list of ``shard_size`` hidden state tensors (on CPU).
    Training data (input_ids) is saved alongside in matching shard files.

    :param model: Teacher model (unquantized) to compute hidden states.
    :param samples: List of input_ids tensors.
    :param hiddens_dir: Directory to write shard files.
    :param shard_size: Number of samples per shard file.
    :return: Total number of shards written.
    """
    hiddens_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    shard_hiddens: list[Tensor] = []
    shard_inputs: list[Tensor] = []

    for data in track(samples, description="Computing teacher hiddens (sharded)"):
        model_input = get_model_input(data)
        hidden = model.model(**model_input).last_hidden_state.cpu()
        shard_hiddens.append(hidden)
        shard_inputs.append(data.cpu())

        if len(shard_hiddens) == shard_size:
            torch.save(
                {"hiddens": shard_hiddens, "input_ids": shard_inputs},
                hiddens_dir / f"shard_{shard_idx:05d}.pt",
            )
            shard_idx += 1
            shard_hiddens = []
            shard_inputs = []

    # Save the last partial shard.
    if shard_hiddens:
        torch.save(
            {"hiddens": shard_hiddens, "input_ids": shard_inputs},
            hiddens_dir / f"shard_{shard_idx:05d}.pt",
        )
        shard_idx += 1

    torch.cuda.empty_cache()
    print(f"Saved {len(samples)} samples in {shard_idx} shards to {hiddens_dir}")
    return shard_idx


class StreamingShardLoader:
    """
    Yields (input_ids_batch, teacher_hiddens_batch) from disk shards.

    Loads one shard at a time into RAM, yields microbatches from it, then moves
    to the next shard. After exhausting all shards, reshuffles shard order and repeats.
    Only ~shard_size * hidden_dim * 2 bytes of hiddens are in RAM at any time.
    """

    def __init__(self, hiddens_dir: Path, microbatch_size: int, device: torch.device, dtype: torch.dtype):
        self.hiddens_dir = hiddens_dir
        self.microbatch_size = microbatch_size
        self.device = device
        self.dtype = dtype
        self.shard_files = sorted(hiddens_dir.glob("shard_*.pt"))
        if not self.shard_files:
            error = f"No shard files found in {hiddens_dir}"
            raise FileNotFoundError(error)
        # Count total samples across all shards (read metadata of first/last to estimate).
        self._total_samples = 0
        for sf in self.shard_files:
            shard = torch.load(sf, weights_only=False, map_location="cpu")
            self._total_samples += len(shard["hiddens"])
            del shard

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yields (input_ids [mb, seqlen], teacher_hiddens [mb, seqlen, hidden]) tuples."""
        while True:
            # Shuffle shard order each pass for better data mixing.
            perm = torch.randperm(len(self.shard_files)).tolist()
            for shard_idx in perm:
                shard = torch.load(self.shard_files[shard_idx], weights_only=False, map_location="cpu")
                hiddens_list = shard["hiddens"]
                inputs_list = shard["input_ids"]
                n = len(hiddens_list)
                # Shuffle samples within the shard.
                sample_perm = torch.randperm(n).tolist()
                for mb_start in range(0, n - n % self.microbatch_size, self.microbatch_size):
                    mb_indices = sample_perm[mb_start : mb_start + self.microbatch_size]
                    input_ids = torch.cat([inputs_list[i] for i in mb_indices], dim=0).to(self.device)
                    teacher_h = torch.cat([hiddens_list[i] for i in mb_indices], dim=0).to(
                        device=self.device, dtype=self.dtype
                    )
                    yield input_ids, teacher_h
                del shard, hiddens_list, inputs_list


# def measure_perplexity(
#     optimum_model: OptimizedModel,
#     max_length: Optional[int] = None,
#     limit: Optional[Union[int, float]] = None,
# ) -> float:
#     """
#     Measure perplexity on the Wikitext dataset, via rolling loglikelihoods for a given model.

#     :param optimum_model: A model to be evaluated.
#     :param max_length: The maximum sequence length for evaluation.
#     :param limit: Limit the number of examples per task (only use this for testing).
#         If <1, limit is a percentage of the total number of examples.
#     :return: The similarity score as a float.
#     """
#     task = "wikitext"
#     print("#" * 50 + " Evaluate via lm-eval-harness " + "#" * 50)
#     lm_obj = OptimumLM(pretrained=optimum_model, max_length=max_length)
#     results = simple_evaluate(lm_obj, tasks=[task], limit=limit, log_samples=False)
#     return results["results"][task]["word_perplexity,none"]


def evaluate_with_vllm(
    checkpoint_dir: Union[str, Path],
    tasks: list[str],
    tensor_parallel_size: int = 2,
    dtype: str = "auto",
    fewshot_as_multiturn: bool = True,
    apply_chat_template: bool = True,
    batch_size: str = "auto",
    limit: Optional[Union[int, float]] = None,
    cuda_devices: str = "1,2",
) -> dict:
    """
    Evaluate a model using lm_eval with vLLM backend in a subprocess.

    Runs evaluation in a subprocess to ensure CUDA_VISIBLE_DEVICES is set before
    CUDA initialization, which is required for proper GPU memory management.

    :param checkpoint_dir: Path to the model checkpoint directory.
    :param tasks: List of evaluation tasks (e.g., ["gsm8k"]).
    :param tensor_parallel_size: Number of GPUs for tensor parallelism.
    :param dtype: Data type for the model (e.g., "auto", "float16", "bfloat16").
    :param fewshot_as_multiturn: Whether to use fewshot examples as multi-turn conversation.
    :param apply_chat_template: Whether to apply the chat template.
    :param batch_size: Batch size for evaluation ("auto" for automatic).
    :param limit: Limit the number of examples per task (only use this for testing).
    :param cuda_devices: Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES.
    :return: Dictionary containing evaluation results.
    """
    print("#" * 50 + " Evaluate via lm-eval-harness (vLLM) " + "#" * 50)

    checkpoint_path = str(checkpoint_dir)
    model_args = f"pretrained={checkpoint_path},dtype={dtype},tensor_parallel_size={tensor_parallel_size}"

    cmd = [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(checkpoint_dir / "lm_eval_results"),
    ]
    if fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    # Set CUDA_VISIBLE_DEVICES in subprocess environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(f"Running: CUDA_VISIBLE_DEVICES={cuda_devices} {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        msg = f"lm_eval failed with return code {result.returncode}"
        raise RuntimeError(msg)

    # Parse results from the output JSON file
    results_dir = checkpoint_dir / "lm_eval_results"
    # Find the most recent results file (lm_eval names them results_<timestamp>.json)
    result_files = list(results_dir.glob("**/results_*.json"))
    if not result_files:
        msg = f"No results_*.json found in {results_dir}"
        raise FileNotFoundError(msg)
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    with open(latest_result) as f:
        results = json.load(f)

    # Print results summary
    print("\nEvaluation Results:")
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if not metric.endswith("_stderr"):
                print(f"  {metric}: {value}")

    return results


def apply_sigmoid_scale_reparam(model: nn.Module, gamma_init: float = 4.0) -> int:
    """
    Apply ApiQ-style sigmoid reparameterization to FQ scale parameters.

    Replaces direct scale training with sigmoid-constrained training:
        scale_effective = sigmoid(gamma) * scale_init

    This bounds the scale to (0, scale_init), preventing scale explosion.
    Initialized at gamma=4.0 so sigmoid(4)≈0.982, i.e. ~98% of original scale.

    The raw _scale_param_storage is updated via a forward pre-hook before each
    quantize() call, so the rest of NNCF (checkpoint save/load, strip) sees
    the effective scale transparently.

    :param model: Model with NNCF FQ+LoRA hooks.
    :param gamma_init: Initial value for gamma parameter. Default 4.0 (sigmoid≈0.982).
    :return: Number of quantizers reparameterized.
    """
    count = 0
    resumed = 0
    hook_storage = get_hook_storage(model)
    for name, module in hook_storage.named_hooks():
        if not isinstance(module, SymmetricLoraQuantizer):
            continue

        if hasattr(module, "_gamma"):
            # Already reparameterized (restored from checkpoint via load_state_dict).
            # Just re-register the hooks (hooks aren't serialized) and reconfigure grads.
            resumed += 1
        else:
            # Fresh init: save original scale as frozen buffer, create gamma.
            scale_init = module._scale_param_storage.data.clone()
            module.register_buffer("_scale_init", scale_init)

            gamma = torch.full_like(scale_init, gamma_init, dtype=torch.float32)
            module._gamma = nn.Parameter(gamma)

        # Keep _scale_param_storage "alive" for gradients: the FQ STE computes
        # ∂L/∂s. We'll chain-rule that into ∂L/∂γ via a backward hook on _gamma.
        # Note: _scale_param_storage stays trainable for STE gradient, but the
        # optimizer never touches it — only _gamma goes into param groups.
        module._scale_param_storage.requires_grad_(True)
        module._gamma.requires_grad_(False)  # Turned on later by set_trainable()

        # Forward pre-hook: update _scale_param_storage.data BEFORE quantize().
        # Uses .data so autograd still treats _scale_param_storage as a leaf.
        def _update_scale(mod, args):
            with torch.no_grad():
                sig = torch.sigmoid(mod._gamma.to(mod._scale_init.dtype))
                mod._scale_param_storage.data.copy_(sig * mod._scale_init)

        module.register_forward_pre_hook(_update_scale)

        # Backward hook on _scale_param_storage: chain-rule gradient to _gamma.
        # ∂L/∂γ = ∂L/∂s · ∂s/∂γ = ∂L/∂s · σ(γ)(1-σ(γ)) · s_init
        # Guard: only register once (tensor hooks survive across forward/backward,
        # but not across save/load — _scale_param_storage is recreated by load_state_dict).
        if not getattr(module, "_sigmoid_bwd_hook_registered", False):

            def _chain_grad_to_gamma(mod):
                def hook(grad):
                    with torch.no_grad():
                        sig = torch.sigmoid(mod._gamma.to(mod._scale_init.dtype))
                        dsig_dgamma = sig * (1.0 - sig)  # sigmoid derivative
                        gamma_grad = (grad * dsig_dgamma * mod._scale_init).to(mod._gamma.dtype)
                        if mod._gamma.grad is None:
                            mod._gamma.grad = gamma_grad
                        else:
                            mod._gamma.grad.add_(gamma_grad)
                    # Zero out scale grad so optimizer (if it somehow sees it) does nothing.
                    return torch.zeros_like(grad)

                return hook

            module._scale_param_storage.register_hook(_chain_grad_to_gamma(module))
            module._sigmoid_bwd_hook_registered = True
        count += 1

    status = f"Sigmoid scale reparameterization: {count} quantizers"
    if resumed:
        status += f" ({resumed} restored from checkpoint)"
    else:
        status += f", gamma_init={gamma_init} (σ={torch.sigmoid(torch.tensor(gamma_init)):.4f})"
    print(status)
    return count


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: list[Tensor]) -> list[Tensor]:
    """
    Calculate the hidden states for each input in the dataloader using the given model.

    :param model: The model used to calculate the hidden states.
    :param dataloader: The dataloader providing the inputs to the model.
    :return: A list of hidden states for each input in the dataloader.
    """
    orig_hiddens = []
    for data in track(dataloader, description="Calculating original hiddens"):
        model_input = get_model_input(data)
        orig_hiddens.append(model.model(**model_input).last_hidden_state.cpu())
    torch.cuda.empty_cache()
    return orig_hiddens


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_hiddens: torch.Tensor, teacher_hiddens: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence loss between the student and teacher hidden states.
    The input tensors are expected to have the same shape, and the last dimension represents the number of classes.

    :param student_hiddens: The hidden states from the student model.
    :param teacher_hiddens: The hidden states from the teacher model.
    :returns: The computed KL divergence loss.
    """
    num_classes = student_hiddens.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, num_classes), dim=-1),
        log_target=True,  # TODO: try without log for teacher with log_target=False
        reduction="batchmean",
    )


def log_quantizer_stats(model: nn.Module, step: int, optimizer: torch.optim.Optimizer) -> None:
    """
    Logs per-quantizer statistics to MLflow: norms of lora_A, lora_B, input_low,
    input_range (asymmetric) or scale (symmetric), gradient norms, and learning rates.

    :param model: The model containing quantizers in its hook storage.
    :param step: Current global training step.
    :param optimizer: The optimizer, used to retrieve per-parameter learning rates.
    """
    # Build a mapping from parameter id to its current learning rate
    param_id_to_lr: dict[int, float] = {}
    for group in optimizer.param_groups:
        lr = group["lr"]
        for p in group["params"]:
            param_id_to_lr[id(p)] = lr

    metrics: dict[str, float] = {}

    # Only log metrics for the first layer encountered at each bit-width to reduce MLflow overhead.
    logged_bits: set[int] = set()
    hook_storage = get_hook_storage(model)
    for name, module in hook_storage.named_hooks():
        if not isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer, StretchedSymmetricLoraQuantizer)):
            continue
        if module.num_bits not in (2, 3, 4):
            continue
        if module.num_bits in logged_bits:
            continue
        logged_bits.add(module.num_bits)

        # Shorten the name for cleaner metric keys (MLflow uses '.' as separator)
        short = name.replace("post_hooks.", "").replace("pre_hooks.", "").replace(".weight__0", "")
        prefix = f"quantizers/{short}"

        # LoRA adapter norms
        metrics[f"{prefix}/lora_A_norm"] = module.lora_A.data.float().norm().item()
        metrics[f"{prefix}/lora_B_norm"] = module.lora_B.data.float().norm().item()

        # Scale / input_low / input_range / alpha norms depending on quantizer type
        if isinstance(module, AsymmetricLoraQuantizer):
            metrics[f"{prefix}/input_low_norm"] = module.input_low.data.float().norm().item()
            metrics[f"{prefix}/input_range_norm"] = module.input_range.data.float().norm().item()
        elif isinstance(module, StretchedSymmetricLoraQuantizer):
            metrics[f"{prefix}/alpha_norm"] = module.alpha.data.float().norm().item()
        elif isinstance(module, SymmetricLoraQuantizer):
            metrics[f"{prefix}/scale_norm"] = module.scale.data.float().norm().item()
            if hasattr(module, "_gamma"):
                gamma_val = module._gamma.data.float()
                metrics[f"{prefix}/gamma_mean"] = gamma_val.mean().item()
                metrics[f"{prefix}/sigmoid_gamma_mean"] = torch.sigmoid(gamma_val).mean().item()

        # Gradient norms
        for param_name, param in [("lora_A", module.lora_A), ("lora_B", module.lora_B)]:
            if param.grad is not None:
                metrics[f"{prefix}/{param_name}_grad_norm"] = param.grad.data.float().norm().item()

        if isinstance(module, AsymmetricLoraQuantizer):
            for param_name, param in [("input_low", module.input_low), ("input_range", module.input_range)]:
                if param.grad is not None:
                    metrics[f"{prefix}/{param_name}_grad_norm"] = param.grad.data.float().norm().item()
        elif isinstance(module, StretchedSymmetricLoraQuantizer):
            s = module._alpha_param_storage
            if s.grad is not None:
                metrics[f"{prefix}/alpha_grad_norm"] = s.grad.data.float().norm().item()
        elif isinstance(module, SymmetricLoraQuantizer):
            if hasattr(module, "_gamma"):
                if module._gamma.grad is not None:
                    metrics[f"{prefix}/gamma_grad_norm"] = module._gamma.grad.data.float().norm().item()
            else:
                s = module._scale_param_storage
                if s.grad is not None:
                    metrics[f"{prefix}/scale_grad_norm"] = s.grad.data.float().norm().item()

        # Learning rates
        for param_name, param in [("lora_A", module.lora_A), ("lora_B", module.lora_B)]:
            lr_val = param_id_to_lr.get(id(param))
            if lr_val is not None:
                metrics[f"{prefix}/{param_name}_lr"] = lr_val

        if isinstance(module, AsymmetricLoraQuantizer):
            for param_name, param in [("input_low", module.input_low), ("input_range", module.input_range)]:
                lr_val = param_id_to_lr.get(id(param))
                if lr_val is not None:
                    metrics[f"{prefix}/{param_name}_lr"] = lr_val
        elif isinstance(module, StretchedSymmetricLoraQuantizer):
            lr_val = param_id_to_lr.get(id(module._alpha_param_storage))
            if lr_val is not None:
                metrics[f"{prefix}/alpha_lr"] = lr_val
        elif isinstance(module, SymmetricLoraQuantizer):
            if hasattr(module, "_gamma"):
                lr_val = param_id_to_lr.get(id(module._gamma))
            else:
                lr_val = param_id_to_lr.get(id(module._scale_param_storage))
            if lr_val is not None:
                metrics[f"{prefix}/scale_lr"] = lr_val

    if metrics:
        mlflow.log_metrics(metrics, step=step)


def set_trainable(
    model: nn.Module,
    lora_lr: float,
    fq_lr: float,
    lora_weight_decay: float = 1e-4,
    fq_weight_decay: float = 0.0,
    tune_bits: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr: Learning rate for the LoRA adapters.
    :param fq_lr: Learning rate for the quantizer scales.
    :param lora_weight_decay: Weight decay for LoRA adapter parameters.
    :param fq_weight_decay: Weight decay for quantizer scale parameters.
    :param tune_bits: List of bit-widths to tune (e.g. [2], [3], [4], or [2, 3, 4]).
        If None, tunes 2, 3, and 4 bit layers.
    :return: A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    if tune_bits is None:
        tune_bits = [2, 3, 4]
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    train_lora = lora_lr > 0
    train_scales = fq_lr > 0
    hook_storage = get_hook_storage(model)
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer, StretchedSymmetricLoraQuantizer)) and (
            module.num_bits in tune_bits
        ):
            # Only enable gradients for param groups that will actually be trained.
            if train_lora:
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
                adapters_to_train.extend(module.get_adapters().values())
            if train_scales:
                if isinstance(module, AsymmetricLoraQuantizer):
                    module.input_low.requires_grad = True
                    module._input_range_param_storage.requires_grad = True
                elif isinstance(module, StretchedSymmetricLoraQuantizer):
                    module._alpha_param_storage.requires_grad = True
                elif isinstance(module, SymmetricLoraQuantizer):
                    if hasattr(module, "_gamma"):
                        # Sigmoid reparameterization: train gamma via chain-rule backward hook.
                        # _scale_param_storage must have requires_grad=True for the STE to
                        # compute ∂L/∂s, which the backward hook chains to ∂L/∂γ.
                        module._scale_param_storage.requires_grad = True
                        module._gamma.requires_grad = True
                        scales_to_train.append(module._gamma)
                    else:
                        module._scale_param_storage.requires_grad = True
                if not (isinstance(module, SymmetricLoraQuantizer) and hasattr(module, "_gamma")):
                    params = module.get_trainable_params()
                    adapters = module.get_adapters()
                    scales_to_train.extend(param for name, param in params.items() if name not in adapters)

    param_groups = []
    if train_lora:
        param_groups.append({"params": adapters_to_train, "lr": lora_lr, "weight_decay": lora_weight_decay})
    if train_scales:
        param_groups.append({"params": scales_to_train, "lr": fq_lr, "weight_decay": fq_weight_decay})

    params = list(model.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    all_param = sum(p.numel() for p in params)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
        f" (lora={'ON' if train_lora else 'OFF'}, scales={'ON' if train_scales else 'OFF'}, "
        f"tune_bits={tune_bits})"
    )
    model.train()
    return param_groups


def save_checkpoint(model: nn.Module, ckpt_file: Path, model_state: bool = True) -> None:
    """
    Stores the current state of a quantized model to a checkpoint file.

    :param model: The model whose state will be saved to checkpoint.
    :param ckpt_file: Path to store the checkpoint file.
    :param model_state: Whether to save the complete model weights in addition to NNCF state. Required when using
        AWQ method which fuses scaling factors into weights. When False, only NNCF configuration and state are saved,
        as they're maintained separately from the model's weights.
    """
    hook_storage = get_hook_storage(model)
    ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf.torch.get_config(model)}
    if model_state:
        ckpt["model_state"] = model.state_dict()
    torch.save(ckpt, ckpt_file)


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


# @torch.no_grad()
# def export_to_openvino(pretrained: str, ckpt_file: Path, ir_dir: Path) -> OVModelForCausalLM:
#     """
#     Create a wrapper of OpenVINO model from the checkpoint for evaluation on CPU via WWB.

#     :param pretrained: The name or path of the pretrained model.
#     :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
#     :param last_dir: The directory where the OpenVINO model will be saved.
#     :return: A wrapper of OpenVINO model ready for evaluation.
#     """
#     model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
#     model_to_eval = load_checkpoint(model_to_eval, ckpt_file)
#     model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.DQ)
#     export_from_model(model_to_eval, ir_dir, device="cpu")
#     return OVModelForCausalLM.from_pretrained(
#         model_id=ir_dir,
#         trust_remote_code=True,
#         load_in_8bit=False,
#         compile=True,
#     )


def limit_type(astr: str):
    value = float(astr)
    if value < 0 or value > 1:
        msg = "value not in range [0,1]"
        raise argparse.ArgumentTypeError(msg)
    return value


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        # default="Qwen/Qwen3-8B",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output",
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank of lora adapters")
    parser.add_argument(
        "--basic_init",
        action="store_true",
        help="Whether to initialize quantization with basic min-max round-to-nearest schema. By default, advanced "
        "data-aware post-training methods are used: AWQ + Scale Estimation. These methods typically provide better "
        "accuracy, but require a calibration dataset and additional initialization time "
        "(~20 sec for 1B and ~80 sec for 8B models).",
    )

    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        default="pile",
        choices=list(DATASET_LOADERS.keys()),
        help="Training dataset to use. 'pile' = NeelNanda/pile-10k, 'wikitext' = Salesforce/wikitext-2-raw-v1. "
        "Default: pile.",
    )
    parser.add_argument("--num_train_samples", type=int, default=512, help="Number of training samples")
    parser.add_argument("--train_seqlen", type=int, default=512, help="Train data context length.")
    parser.add_argument("--eval_seqlen", type=int, default=2048, help="Evaluation data context length.")
    parser.add_argument(
        "--limit",
        type=limit_type,
        default=None,
        help="A percentage of the total number of examples for evaluation. "
        "Should be on the range [0,1]. If None, all samples will be used.",
    )

    # Training params
    parser.add_argument(
        "--fq_lr",
        type=float,
        default=1e-3,
        help="Learning rate for quantizer scales (input_low, input_range, scale). Set to 0 to freeze scales entirely.",
    )
    parser.add_argument(
        "--lora_lr",
        type=float,
        default=0.0,
        help="Learning rate for LoRA adapters (lora_A, lora_B). "
        "Set to 0 to freeze LoRA (default). Typical values: 1e-5 to 1e-4.",
    )
    parser.add_argument(
        "--fq_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization) for quantizer scale parameters. 0 disables.",
    )
    parser.add_argument(
        "--lora_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) for LoRA adapter parameters. 0 disables.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=4,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    parser.add_argument(
        "--constant_epochs",
        type=int,
        default=0,
        help="Number of epochs with constant LR (always the first phase). "
        "If >0 and cosine_epochs==0, training uses only constant LR.",
    )
    parser.add_argument(
        "--cosine_epochs",
        type=int,
        default=1,
        help="Number of epochs with cosine annealing LR (always the second phase). "
        "If >0 and constant_epochs==0, training uses only cosine annealing.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.0,
        help="Minimum LR at end of cosine annealing, expressed as a fraction of the initial LR. "
        "0.0 means LR decays to zero; 0.1 means it decays to 10%% of peak. "
        "Ignored when cosine_epochs==0.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=0.0,
        help="Number of epochs for linear warmup (LR ramps from 0 to peak). "
        "Supports fractional values, e.g. 0.5 for half an epoch. 0 disables warmup.",
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 15],
        help="Epoch numbers (0-indexed) at which to save a checkpoint. "
        "Epoch 0 means after initialization (before any training). "
        "The last epoch always saves regardless of this list.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name. If not specified, uses a timestamp.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: uses a separate MLflow database (mlflow_debug.db) and experiment prefix "
        "so that experimental runs do not pollute production results.",
    )
    parser.add_argument(
        "--mlflow_db",
        type=str,
        default=None,
        help="Override the MLflow SQLite database path. "
        "By default: <output_dir>/mlflow.db (or mlflow_debug.db in --debug mode).",
    )
    parser.add_argument(
        "--compression_format",
        type=str,
        default="FQ_STRETCHED_LORA",
        choices=[f.name for f in CompressionFormat],
        help="Compression format to use. Key options: "
        "FQ_LORA (standard fake-quantize + LoRA), "
        "FQ_STRETCHED_LORA (ParetoQ-style stretched quantization + LoRA). "
        "Default: FQ_STRETCHED_LORA.",
    )
    parser.add_argument(
        "--init_ckpt",
        type=Path,
        default=None,
        help="Path to the initial compression checkpoint (.pth) to resume from. "
        "If not specified, auto-derived from --compression_format: "
        "<output_dir>/nncf_init_<format>.pth (e.g. nncf_init_fq_stretched_lora.pth). "
        "This allows different formats to maintain separate init checkpoints.",
    )
    parser.add_argument(
        "--use_autograd_quantize",
        action="store_true",
        help="Use STE-based autograd for gradient computation instead of hand-written backward.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade compute for memory. "
        "Recomputes activations during backward instead of storing them, "
        "reducing peak GPU memory at the cost of ~30%% slower training.",
    )
    parser.add_argument(
        "--tune_bits",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Which bit-width layers to tune. Use for staged tuning: "
        "first run with --tune_bits 2 to tune only 2-bit layers, "
        "then run with --tune_bits 4 --init_ckpt <2bit_ckpt> to tune only 4-bit layers "
        "while keeping 2-bit layers frozen. Default: [2, 3, 4] (tune all).",
    )
    parser.add_argument(
        "--sigmoid_scale",
        action="store_true",
        help="Apply ApiQ-style sigmoid reparameterization to FQ scales: "
        "scale = sigmoid(gamma) * scale_init. Constrains scale to (0, scale_init), "
        "preventing explosion. Gamma initialized at 4.0 (sigmoid≈0.982). "
        "More stable than direct scale training, especially with higher LRs.",
    )

    # ── Step-based streaming training (LR-QAT style: unique data per step) ──
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total number of optimizer steps. When set, enables streaming mode: "
        "pre-computes teacher hiddens for --num_train_samples into disk shards, "
        "then streams through them for --total_steps steps (unique data each step, "
        "cycling with reshuffle when pool is exhausted). "
        "Epoch-based args (--constant_epochs, --cosine_epochs, etc.) are ignored.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of linear warmup steps (streaming mode only). 0 disables warmup.",
    )
    parser.add_argument(
        "--constant_steps",
        type=int,
        default=0,
        help="Number of constant-LR steps after warmup (streaming mode only). "
        "Remaining steps after warmup + constant use cosine annealing.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 5000, 10000],
        help="Step numbers at which to save a checkpoint (streaming mode only). "
        "The final step always saves regardless of this list.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=512,
        help="Number of samples per shard file for sharded hiddens (streaming mode only). "
        "Smaller = less RAM but more disk I/O. Default: 512.",
    )
    parser.add_argument(
        "--hiddens_dir",
        type=str,
        default=None,
        help="Path to a directory with pre-computed sharded teacher hiddens. "
        "If provided, skips hiddens precomputation and reuses the shards from this path. "
        "Useful for re-running experiments with different hyperparameters without "
        "recomputing teacher hiddens (e.g. from a previous run's output_dir/hiddens_shards).",
    )
    return parser


def main(argv) -> int:
    """
    Fine-tunes the specified model and returns 0 on success or 1 on failure.
    Errors are caught and printed so that multiple sequential runs (e.g. from shell scripts)
    are not interrupted by a single failure.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    try:
        return _main_impl(args)
    except Exception:
        traceback.print_exc()
        print(f"\n{'!' * 60}")
        print(f"RUN FAILED: {args.run_name or 'unnamed'}")
        print(f"{'!' * 60}\n")
        # Clean up GPU memory so the next run can start fresh.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 1


def _main_impl(args) -> int:
    """Core implementation of main(). Raises on error."""
    if args.fq_lr == 0 and args.lora_lr == 0:
        print("Both --fq_lr and --lora_lr are 0 — nothing to train. Skipping.")
        return 0
    assert torch.cuda.is_available()
    transformers.set_seed(42)
    set_use_autograd_quantize(args.use_autograd_quantize)
    device = "cuda"
    torch_dtype = torch.bfloat16
    compression_config = dict(
        mode=CompressWeightsMode.INT3_SYM,
        group_size=64,
        awq=not args.basic_init,
        backup_mode=nncf.BackupMode.INT8_SYM,
        scale_estimation=not args.basic_init,
        compression_format=CompressionFormat[args.compression_format],
    )
    pprint({"CLI arguments": vars(args), "Major compression parameters": compression_config})
    compression_config["advanced_parameters"] = AdvancedCompressionParameters(
        awq_params=AdvancedAWQParameters(prefer_data_aware_scaling=not args.basic_init),
        # scale_estimation_params=AdvancedScaleEstimationParameters(subset_size=-1, initial_steps=10, scale_steps=10),
        lora_adapter_rank=args.lora_rank,
    )
    # Configure output and log files.
    output_dir = Path(args.output_dir)
    last_dir = output_dir / "last"
    streaming_mode = args.total_steps is not None
    if not streaming_mode:
        # Epoch-based mode: always start fresh.
        shutil.rmtree(last_dir, ignore_errors=True)
    # Streaming mode: keep last_dir intact so auto-resume can find existing checkpoints.
    for path in [output_dir, last_dir]:
        path.mkdir(exist_ok=True, parents=True)
    # Derive initial compression checkpoint path from format (or use explicit override).
    if args.init_ckpt is not None:
        ckpt_file = args.init_ckpt
    else:
        fmt_tag = args.compression_format.lower()
        ckpt_file = output_dir / f"nncf_init_{fmt_tag}.pth"
    hidden_file = output_dir / "hiddens.pth"

    # Configure MLflow tracking (SQLite backend for indexing & future-proofing; file store is deprecated).
    if args.mlflow_db:
        db_path = Path(args.mlflow_db).resolve()
    else:
        db_name = "mlflow_debug.db" if args.debug else "mlflow.db"
        db_path = output_dir.resolve() / db_name
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_prefix = "debug_" if args.debug else ""
    mlflow.set_experiment(f"{experiment_prefix}fqlora_{Path(args.pretrained).name}")
    print(f"MLflow tracking URI: {tracking_uri}  (run `mlflow ui --backend-store-uri {tracking_uri}`)")
    if args.debug:
        print("  ** DEBUG MODE — results go to a separate database and experiment **")
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    with mlflow.start_run(run_name=run_name):
        if streaming_mode:
            _train_streaming(args, compression_config, device, torch_dtype, last_dir, output_dir, ckpt_file)
        else:
            _train(args, compression_config, device, torch_dtype, last_dir, output_dir, ckpt_file, hidden_file)

        # Free GPU memory before loading checkpoints for stripping & evaluation.
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # ── Build evaluation list: init checkpoint + trained checkpoints ──
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        eval_checkpoints: list[tuple[int, Path]] = [(0, ckpt_file)]
        if streaming_mode:
            step_ckpts = sorted(
                last_dir.glob("nncf_checkpoint_step*.pth"),
                key=lambda p: int(p.stem.replace("nncf_checkpoint_step", "")),
            )
            for p in step_ckpts:
                eval_checkpoints.append((int(p.stem.replace("nncf_checkpoint_step", "")), p))
        else:
            epoch_ckpts = sorted(
                last_dir.glob("nncf_checkpoint_epoch*.pth"),
                key=lambda p: int(p.stem.replace("nncf_checkpoint_epoch", "")),
            )
            for p in epoch_ckpts:
                eval_checkpoints.append((int(p.stem.replace("nncf_checkpoint_epoch", "")), p))

        for ckpt_num, ckpt_path in eval_checkpoints:
            # Cache eval results only for ckpt_num 0 (initial PTQ checkpoint) — reusable across runs.
            # Trained checkpoints are always re-evaluated (caching would be error-prone with tuning).
            cache_file = ckpt_path.with_suffix(".eval.json") if ckpt_num == 0 else None
            if cache_file and cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                lambada_acc = cached["lambada_acc"]
                lambada_ppl = cached["lambada_ppl"]
                print(
                    f"Checkpoint {ckpt_num} — cached from {cache_file.name}: "
                    f"acc={lambada_acc:.4f}, ppl={lambada_ppl:.4f}"
                )
            else:
                stripped_dir = last_dir / "stripped"
                print(f"\n{'=' * 60}")
                print(f"Stripping & evaluating: {ckpt_path.name} (checkpoint {ckpt_num})")
                print(f"{'=' * 60}")

                model_to_strip = AutoModelForCausalLM.from_pretrained(
                    args.pretrained, torch_dtype=torch_dtype, device_map="cpu"
                )
                model_to_strip = load_checkpoint(model_to_strip, ckpt_path)
                model_to_strip = nncf.strip(model_to_strip, strip_format=nncf.StripFormat.IN_PLACE)
                if stripped_dir.exists():
                    shutil.rmtree(stripped_dir)
                model_to_strip.save_pretrained(stripped_dir)
                tokenizer.save_pretrained(stripped_dir)
                del model_to_strip
                gc.collect()
                torch.cuda.empty_cache()

                eval_results = evaluate_with_vllm(
                    checkpoint_dir=stripped_dir,
                    tasks=["lambada_openai"],
                    tensor_parallel_size=2,
                    dtype="auto",
                    fewshot_as_multiturn=False,
                    cuda_devices="1,2",
                    apply_chat_template=False,
                    batch_size="auto",
                    limit=args.limit,
                )
                lambada_acc = eval_results["results"]["lambada_openai"]["acc,none"]
                lambada_ppl = eval_results["results"]["lambada_openai"]["perplexity,none"]
                if cache_file:
                    with open(cache_file, "w") as f:
                        json.dump({"lambada_acc": lambada_acc, "lambada_ppl": lambada_ppl}, f, indent=2)

            mlflow.log_metrics(
                {"lambada_acc": lambada_acc, "lambada_ppl": lambada_ppl},
                step=ckpt_num,
            )
            print(f"Checkpoint {ckpt_num} — LAMBADA accuracy: {lambada_acc:.4f}, perplexity: {lambada_ppl:.4f}")

    # del model
    # Export the best tuned model to OpenVINO and evaluate it using LM-Evaluation-Harness.
    # model_for_eval = export_to_openvino(args.pretrained, ckpt_file, ckpt_file.parent)
    # ov_perplexity = measure_perplexity(model_for_eval, args.eval_seqlen, args.limit)
    # mlflow.log_metric("ov_perplexity", ov_perplexity, step=0)
    # print(
    #     f"The finetuned model has been exported to OpenVINO and saved to: {last_dir}\n"
    #     f"The word perplexity on wikitext (test) = {ov_perplexity:.4f}"
    # )
    # return ov_perplexity
    # return gsm8k_acc
    return 0


def _train(args, compression_config, device, torch_dtype, last_dir, output_dir, ckpt_file, hidden_file):
    """Load model, prepare data, run distillation QAT, strip, and save. All heavy objects are local."""
    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Prepare training and calibration data
    load_fn = DATASET_LOADERS[args.dataset]
    train_loader = load_fn(
        num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    )
    if args.basic_init:
        example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
        dataset = Dataset([example_input])
    else:
        calib_loader = load_fn(num_samples=128, seqlen=128, tokenizer=tokenizer, device=device)
        dataset = Dataset(map(get_model_input, calib_loader))

    # Pre-compute hiddens of teacher model for distillation loss.
    if hidden_file.exists():
        orig_hiddens = torch.load(hidden_file, weights_only=False, map_location="cpu")
    else:
        orig_hiddens = calc_hiddens(model, train_loader)
        torch.save(orig_hiddens, hidden_file)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    if ckpt_file.exists():
        print(f"Loading existing init checkpoint: {ckpt_file}")
        model = load_checkpoint(model, ckpt_file)
    else:
        model = compress_weights(model, dataset=dataset, **compression_config)
        save_checkpoint(model, ckpt_file, model_state=False)
        print(f"Saved init checkpoint: {ckpt_file}")

    if args.sigmoid_scale:
        apply_sigmoid_scale_reparam(model)

    # Enable gradient checkpointing to reduce activation memory.
    # ForwardWithHooks keeps FunctionHookMode alive across forward+backward so that
    # checkpoint recomputation sees the same FQ hooks as the original forward.
    # use_reentrant=False is preferred: it counts saved tensors and the persistent
    # mode ensures the counts match between forward and recomputation.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled (use_reentrant=False)")

    param_to_train = set_trainable(
        model,
        lora_lr=args.lora_lr,
        fq_lr=args.fq_lr,
        lora_weight_decay=args.lora_weight_decay,
        fq_weight_decay=args.fq_weight_decay,
        tune_bits=args.tune_bits,
    )
    opt = torch.optim.AdamW(param_to_train, betas=(0.9, 0.95))

    # Run tuning with distillation loss and validation after each epoch.
    args.epochs = args.constant_epochs + args.cosine_epochs
    assert args.epochs > 0, "At least one of --constant_epochs or --cosine_epochs must be > 0."
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    # Compute step counts from total microbatches (not per-epoch * epochs) because
    # grad_steps carries over across epoch boundaries when microbatches don't divide evenly.
    total_training_steps = (microbatches_per_epoch * args.epochs) // grad_accumulation_steps
    constant_steps = (microbatches_per_epoch * args.constant_epochs) // grad_accumulation_steps
    cosine_steps = total_training_steps - constant_steps

    # Build LR scheduler: [linear warmup] -> [constant phase] -> [cosine annealing phase].
    # Uses LambdaLR so that cosine smoothly decays from 1.0 to min_lr_ratio per param group
    # (no floor-clipping artifacts).
    warmup_steps = int(microbatches_per_epoch * args.warmup_epochs) // grad_accumulation_steps

    def _make_lr_lambda(warmup_steps: int, constant_steps: int, cosine_steps: int, min_lr_ratio: float):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            adjusted_step = step - warmup_steps
            if adjusted_step < constant_steps:
                return 1.0
            if cosine_steps <= 0:
                return 1.0
            progress = (adjusted_step - constant_steps) / cosine_steps
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

        return lr_lambda

    scheduler = LambdaLR(opt, lr_lambda=_make_lr_lambda(warmup_steps, constant_steps, cosine_steps, args.min_lr_ratio))

    print(
        f"LR schedule: {warmup_steps} warmup steps, "
        f"{args.constant_epochs} constant epoch(s) ({constant_steps} steps), "
        f"then {args.cosine_epochs} cosine epoch(s) ({cosine_steps} steps, min_lr_ratio={args.min_lr_ratio}). "
        f"Total: {args.epochs} epoch(s), {total_training_steps} steps."
    )

    # Log all meaningful parameters to MLflow.
    mlflow.log_params(
        {
            "pretrained": args.pretrained,
            "lora_rank": args.lora_rank,
            "basic_init": args.basic_init,
            "num_train_samples": args.num_train_samples,
            "train_seqlen": args.train_seqlen,
            "fq_lr": args.fq_lr,
            "lora_lr": args.lora_lr,
            "weight_decay_lora": args.lora_weight_decay,
            "weight_decay_fq": args.fq_weight_decay,
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "grad_accumulation_steps": grad_accumulation_steps,
            "constant_epochs": args.constant_epochs,
            "cosine_epochs": args.cosine_epochs,
            "total_epochs": args.epochs,
            "min_lr_ratio": args.min_lr_ratio,
            "warmup_epochs": args.warmup_epochs,
            "warmup_steps": warmup_steps,
            "constant_steps": constant_steps,
            "cosine_steps": cosine_steps,
            "total_training_steps": total_training_steps,
            "compression_mode": str(compression_config["mode"]),
            "group_size": compression_config["group_size"],
            "compression_format": str(compression_config["compression_format"]),
            "awq": compression_config["awq"],
            "scale_estimation": compression_config["scale_estimation"],
            "use_autograd_quantize": args.use_autograd_quantize,
            "gradient_checkpointing": args.gradient_checkpointing,
            "tune_bits": args.tune_bits,
            "dataset": args.dataset,
        }
    )

    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_steps = 0
    save_epochs = set(args.save_epochs)

    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
            indices = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            # Compute distillation loss between logits of the original model and the model with FQ + LoRA.
            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                targets = model.lm_head(form_batch(orig_hiddens, model_input=False))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma has post-processing after lm_head
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls
            outputs = model(**inputs).logits
            loss = kl_div(outputs, targets.to(dtype=torch_dtype, device=device))

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()
            if grad_steps == grad_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                scheduler.step()
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
                total_steps += 1
                mlflow.log_metric("loss", aggregated_loss, step=total_steps)
                log_quantizer_stats(model, total_steps, opt)
                opt.zero_grad()
        # Save checkpoint at scheduled epochs (epoch+1 because we just finished this epoch).
        finished_epoch = epoch + 1
        if finished_epoch in save_epochs or epoch == args.epochs - 1:
            ckpt_name = f"nncf_checkpoint_epoch{finished_epoch}.pth"
            save_checkpoint(model, last_dir / ckpt_name, model_state=False)


def _train_streaming(args, compression_config, device, torch_dtype, last_dir, output_dir, ckpt_file):
    """
    Step-based streaming training (LR-QAT style).

    Pre-computes teacher hiddens for a large pool of samples and saves them to disk shards.
    Then streams through shards during training, so each optimizer step sees (mostly) unique data.
    The pool is cycled through with reshuffling when exhausted.

    Key differences from _train():
    - No epoch concept — purely step-based (--total_steps).
    - Teacher hiddens stored on disk in shards, loaded one shard at a time (~3 GB RAM).
    - LR schedule specified directly in steps (--warmup_steps, --constant_steps).
    - Much larger data pool without RAM/GPU OOM (limited only by disk space and precompute time).
    """
    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # ── Prepare calibration data for compress_weights (small set, same as before) ──
    if args.basic_init:
        example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
        dataset = Dataset([example_input])
    else:
        load_fn = DATASET_LOADERS[args.dataset]
        calib_loader = load_fn(num_samples=128, seqlen=128, tokenizer=tokenizer, device=device)
        dataset = Dataset(map(get_model_input, calib_loader))

    # ── Pre-compute sharded teacher hiddens ──
    if args.hiddens_dir is not None:
        hiddens_dir = Path(args.hiddens_dir)
        if not (hiddens_dir / "_DONE").exists():
            error = (
                f"--hiddens_dir={hiddens_dir} does not contain a _DONE marker. "
                "Make sure hiddens were fully pre-computed."
            )
            raise FileNotFoundError(error)
        print(f"Reusing pre-computed hiddens from {hiddens_dir}")
    else:
        hiddens_dir = output_dir / "hiddens_shards"
    done_marker = hiddens_dir / "_DONE"
    if done_marker.exists():
        print(f"Sharded hiddens already exist in {hiddens_dir}, skipping precompute.")
    else:
        print(f"Pre-computing teacher hiddens for {args.num_train_samples} samples (shard_size={args.shard_size})...")
        # Collect training samples from the dataset.
        load_fn = DATASET_LOADERS[args.dataset]
        train_samples = load_fn(
            num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
        )
        num_shards = calc_hiddens_sharded(model, train_samples, hiddens_dir, shard_size=args.shard_size)
        del train_samples
        torch.cuda.empty_cache()
        done_marker.touch()
        print(f"Teacher hiddens: {num_shards} shards written to {hiddens_dir}")

    # ── Auto-detect latest training checkpoint for resume ──
    resume_step = 0
    resume_training_state = None
    total_training_steps = args.total_steps

    # Scan last_dir for step checkpoints with step < total_steps.
    step_ckpts = sorted(
        last_dir.glob("nncf_checkpoint_step*.pth"),
        key=lambda p: int(p.stem.replace("nncf_checkpoint_step", "")),
    )
    resume_ckpt_path = None
    for p in reversed(step_ckpts):
        step_num = int(p.stem.replace("nncf_checkpoint_step", ""))
        if step_num < total_training_steps:
            resume_ckpt_path = p
            break

    if resume_ckpt_path is not None:
        print(f"Found training checkpoint: {resume_ckpt_path}")
        resume_ckpt_data = torch.load(resume_ckpt_path, weights_only=False, map_location="cpu")
        if "training_state" not in resume_ckpt_data:
            print("  WARNING: checkpoint has no 'training_state' — cannot resume, starting fresh.")
        else:
            resume_training_state = resume_ckpt_data["training_state"]
            resume_step = resume_training_state["step"]
            print(f"  Will resume from step {resume_step}")
        del resume_ckpt_data
        model = load_checkpoint(model, resume_ckpt_path)
    elif ckpt_file.exists():
        print(f"Loading existing init checkpoint: {ckpt_file}")
        model = load_checkpoint(model, ckpt_file)
    else:
        model = compress_weights(model, dataset=dataset, **compression_config)
        save_checkpoint(model, ckpt_file, model_state=False)
        print(f"Saved init checkpoint: {ckpt_file}")

    if args.sigmoid_scale:
        apply_sigmoid_scale_reparam(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled (use_reentrant=False)")

    param_to_train = set_trainable(
        model,
        lora_lr=args.lora_lr,
        fq_lr=args.fq_lr,
        lora_weight_decay=args.lora_weight_decay,
        fq_weight_decay=args.fq_weight_decay,
        tune_bits=args.tune_bits,
    )
    opt = torch.optim.AdamW(param_to_train, betas=(0.9, 0.95))

    # ── LR schedule: warmup → constant → cosine (all in steps) ──
    w_steps = args.warmup_steps
    c_steps = args.constant_steps
    cosine_steps = max(0, total_training_steps - w_steps - c_steps)

    def _make_lr_lambda(warmup_steps: int, constant_steps: int, cosine_steps: int, min_lr_ratio: float):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            adjusted_step = step - warmup_steps
            if adjusted_step < constant_steps:
                return 1.0
            if cosine_steps <= 0:
                return 1.0
            progress = (adjusted_step - constant_steps) / cosine_steps
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

        return lr_lambda

    scheduler = LambdaLR(opt, lr_lambda=_make_lr_lambda(w_steps, c_steps, cosine_steps, args.min_lr_ratio))

    # ── Restore optimizer/scheduler state when resuming ──
    if resume_training_state is not None:
        opt.load_state_dict(resume_training_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_training_state["scheduler_state_dict"])
        del resume_training_state
        print(f"Restored optimizer & scheduler state (resuming from step {resume_step})")

    # ── Create streaming data loader ──
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    shard_loader = StreamingShardLoader(hiddens_dir, args.microbatch_size, device, torch_dtype)

    print(
        f"Streaming training: {total_training_steps} steps, "
        f"{shard_loader.total_samples} unique samples in pool, "
        f"batch={args.batch_size}, microbatch={args.microbatch_size}, "
        f"grad_accum={grad_accumulation_steps}.\n"
        f"LR schedule: {w_steps} warmup → {c_steps} constant → {cosine_steps} cosine "
        f"(min_lr_ratio={args.min_lr_ratio}).\n"
        f"Samples per full pass: {shard_loader.total_samples}, "
        f"steps per pass: {shard_loader.total_samples // args.batch_size}, "
        f"expected passes: {total_training_steps * args.batch_size / max(1, shard_loader.total_samples):.1f}x."
    )

    # Log all meaningful parameters to MLflow.
    mlflow.log_params(
        {
            "pretrained": args.pretrained,
            "lora_rank": args.lora_rank,
            "basic_init": args.basic_init,
            "num_train_samples": args.num_train_samples,
            "train_seqlen": args.train_seqlen,
            "fq_lr": args.fq_lr,
            "lora_lr": args.lora_lr,
            "weight_decay_lora": args.lora_weight_decay,
            "weight_decay_fq": args.fq_weight_decay,
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "grad_accumulation_steps": grad_accumulation_steps,
            "total_training_steps": total_training_steps,
            "warmup_steps": w_steps,
            "constant_steps": c_steps,
            "cosine_steps": cosine_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "compression_mode": str(compression_config["mode"]),
            "group_size": compression_config["group_size"],
            "compression_format": str(compression_config["compression_format"]),
            "awq": compression_config["awq"],
            "scale_estimation": compression_config["scale_estimation"],
            "use_autograd_quantize": args.use_autograd_quantize,
            "gradient_checkpointing": args.gradient_checkpointing,
            "tune_bits": args.tune_bits,
            "dataset": args.dataset,
            "streaming_mode": True,
            "shard_size": args.shard_size,
            "total_unique_samples": shard_loader.total_samples,
        }
    )

    # ── Training loop ──
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = 0
    total_steps = resume_step
    save_step_set = set(args.save_steps)
    data_iter = iter(shard_loader)
    remaining_microbatches = (total_training_steps - resume_step) * grad_accumulation_steps

    if resume_step > 0:
        print(f"Resuming training from step {resume_step} ({remaining_microbatches} microbatches remaining)")

    for microbatch_idx in track(range(remaining_microbatches), description="Training"):
        input_ids, teacher_hiddens = next(data_iter)

        # Compute teacher targets via lm_head (same as epoch-based training).
        model_input = get_model_input(input_ids)
        with torch.no_grad():
            targets = model.lm_head(teacher_hiddens)
            if hasattr(model.config, "final_logit_softcapping"):
                fls = model.config.final_logit_softcapping
                if fls is not None:
                    targets = targets / fls
                    targets = torch.tanh(targets)
                    targets = targets * fls

        outputs = model(**model_input).logits
        loss = kl_div(outputs, targets.to(dtype=torch_dtype, device=device))

        loss_numerator += loss.item()
        grad_steps += 1
        if not torch.isfinite(loss).item():
            err = f"Fine-tuning loss is {loss}"
            raise ValueError(err)
        (loss / grad_accumulation_steps).backward()

        if grad_steps == grad_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            aggregated_loss = loss_numerator / grad_steps
            loss_numerator = grad_steps = 0
            total_steps += 1
            mlflow.log_metric("loss", aggregated_loss, step=total_steps)
            log_quantizer_stats(model, total_steps, opt)
            opt.zero_grad()

            # Save checkpoint at scheduled steps (includes optimizer/scheduler for resume).
            if total_steps in save_step_set or total_steps == total_training_steps:
                ckpt_name = f"nncf_checkpoint_step{total_steps}.pth"
                ckpt_path = last_dir / ckpt_name
                save_checkpoint(model, ckpt_path, model_state=False)
                # Append training state for resume support.
                ckpt_data = torch.load(ckpt_path, weights_only=False, map_location="cpu")
                ckpt_data["training_state"] = {
                    "step": total_steps,
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(ckpt_data, ckpt_path)
                print(f"  [step {total_steps}] saved {ckpt_name}, loss={aggregated_loss:.6f}")

            if total_steps >= total_training_steps:
                break


if __name__ == "__main__":
    main(sys.argv[1:])
